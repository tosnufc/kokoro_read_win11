import os
import re
import sys
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from kokoro_onnx import Kokoro
    import soundfile as sf
    import pyaudio
    import numpy as np
    import pyperclip
    import psutil
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements with: pip install -r requirements.txt")
    sys.exit(1)

class OptimizedTTSReader:
    def __init__(self):
        self.sample_rate = 24000
        self.chunk_size = 1024
        self.output_dir = Path('Output_audio')
        self.kokoro = None
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Streaming settings
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.is_playing = False
        self.is_processing = False
        
        print(f"🚀 Streaming TTS: Process while playing")
        print(f"🖥️ System: {os.cpu_count() or 4} cores, FP16 model")
        
    def optimize_system(self):
        """Apply Linux-style ONNX optimizations"""
        try:
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
            
            # Linux-style ONNX optimizations that work great on Windows
            os.environ['ORT_NUM_THREADS'] = str(os.cpu_count() or 4)
            os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
            os.environ['ORT_ENABLE_CPU_FP16_OPS'] = '1'
            os.environ['ORT_ENABLE_ALL_OPTIMIZATIONS'] = '1'
            os.environ['ORT_DISABLE_TRT_FLASH_ATTENTION'] = '0'
            os.environ['ORT_ENABLE_MEMORY_ARENA_SHRINKAGE'] = '1'
            
            # Additional Linux-style optimizations
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['ORT_THREAD_SPINNING_POLICY'] = '1'
            
            return True
        except Exception:
            return False
        
    def download_fp16_model(self):
        """Download FP16 model (proven fastest)"""
        model_dir = Path("model")
        voice_dir = Path("voice")
        model_dir.mkdir(exist_ok=True)
        voice_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "kokoro-v1.0.fp16.onnx"
        voices_path = voice_dir / "voices-v1.0.bin"
        
        if model_path.exists() and voices_path.exists():
            return str(model_path), str(voices_path)
        
        print("📥 Downloading FP16 model (169MB)...")
        
        try:
            import urllib.request
            
            if not model_path.exists():
                urllib.request.urlretrieve(
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx",
                    str(model_path)
                )
            
            if not voices_path.exists():
                urllib.request.urlretrieve(
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                    str(voices_path)
                )
            
            return str(model_path), str(voices_path)
        except Exception:
            return None, None

    def initialize_tts(self):
        """Initialize TTS with optimal settings"""
        self.optimize_system()
        
        model_path, voices_path = self.download_fp16_model()
        if not model_path or not voices_path:
            return False
        
        print("⚡ Loading FP16 model...")
        self.kokoro = Kokoro(model_path, voices_path)
        print("✅ Ready!")
        return True
    
    def split_into_sentences(self, text):
        """Split text into sentences for streaming"""
        if not text or not text.strip():
            return []
        
        # Split by sentence endings, preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 1:  # Skip empty or single character sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def generate_sentence_audio(self, sentence, voice='af_heart', speed=1.2):
        """Generate audio for one sentence"""
        if not sentence.strip() or self.kokoro is None:
            return None
        
        try:
            samples, sample_rate = self.kokoro.create(sentence, voice=voice, speed=speed, lang="en-us")
            return (samples, sample_rate)
        except Exception as e:
            print(f"❌ Error generating audio for sentence: {e}")
            return None
    
    def audio_processor(self, sentences, voice='af_heart', speed=1.2):
        """Process sentences and add to audio queue"""
        self.is_processing = True
        print(f"🔄 Processing {len(sentences)} sentences...")
        
        for i, sentence in enumerate(sentences):
            if not self.is_processing:  # Check if we should stop
                break
                
            print(f"⚡ Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            audio_data = self.generate_sentence_audio(sentence, voice, speed)
            if audio_data:
                self.audio_queue.put((i, audio_data))
                print(f"✅ Sentence {i+1} ready for playback")
            else:
                print(f"❌ Failed to process sentence {i+1}")
        
        # Signal end of processing
        self.audio_queue.put(None)
        self.is_processing = False
        print("🏁 Processing complete")
    
    def audio_player(self):
        """Play audio from queue"""
        self.is_playing = True
        print("🔊 Starting audio playback...")
        
        if not self.initialize_audio():
            print("❌ Failed to initialize audio")
            self.is_playing = False
            return
        
        try:
            while self.is_playing:
                try:
                    # Get audio data from queue with timeout
                    item = self.audio_queue.get(timeout=1.0)
                    
                    if item is None:  # End signal
                        break
                    
                    sentence_idx, audio_data = item
                    samples, sample_rate = audio_data
                    
                    print(f"🎵 Playing sentence {sentence_idx + 1}...")
                    
                    # Play the audio
                    audio_data_int16 = (np.asarray(samples) * 32767).astype(np.int16)
                    chunk_size = self.chunk_size * 4
                    
                    if self.audio_stream is not None:
                        for i in range(0, len(audio_data_int16), chunk_size):
                            if not self.is_playing:  # Check if we should stop
                                break
                            chunk = audio_data_int16[i:i + chunk_size]
                            self.audio_stream.write(chunk.tobytes())
                    
                    print(f"✅ Sentence {sentence_idx + 1} played")
                    
                except queue.Empty:
                    # No audio data available, check if processing is still ongoing
                    if not self.is_processing:
                        break
                    continue
                    
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
        finally:
            self.cleanup_audio()
            self.is_playing = False
            print("🔇 Audio playback stopped")
    
    def initialize_audio(self):
        """Initialize audio"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            default_device = self.pyaudio_instance.get_default_output_device_info()
            
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size * 8,
                output_device_index=int(default_device['index'])
            )
            return True
        except Exception:
            return False
    
    def cleanup_audio(self):
        """Cleanup audio"""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
        except Exception:
            pass
    
    def save_audio(self, audio_data, filename='output.wav'):
        """Save audio to file"""
        try:
            samples, sample_rate = audio_data
            
            self.output_dir.mkdir(exist_ok=True)
            output_path = self.output_dir / filename
            
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples * (0.95 / max_val)
            
            sf.write(str(output_path), samples, sample_rate)
            print(f"💾 Saved: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return False
    
    def run(self):
        """Main execution with streaming"""
        print("=== Streaming TTS ===")
        print("🎯 Process while playing strategy")
        print()
        
        try:
            # Get text
            text = pyperclip.paste()
            if not text or not text.strip():
                print("❌ No text in clipboard")
                return False
            
            print(f"✅ Text: {len(text)} characters")
            print()
            
            # Initialize TTS
            if not self.initialize_tts():
                return False
            
            # Split into sentences
            sentences = self.split_into_sentences(text)
            if not sentences:
                print("❌ No valid sentences found")
                return False
            
            print(f"📝 Found {len(sentences)} sentences")
            print()
            
            # Start processing and playing threads
            start_time = time.time()
            
            # Start audio processor thread
            processor_thread = threading.Thread(
                target=self.audio_processor, 
                args=(sentences, 'af_heart', 1.2)
            )
            processor_thread.daemon = True
            processor_thread.start()
            
            # Start audio player thread
            player_thread = threading.Thread(target=self.audio_player)
            player_thread.daemon = True
            player_thread.start()
            
            # Wait for both threads to complete
            processor_thread.join()
            player_thread.join()
            
            total_time = time.time() - start_time
            print(f"⏱️ Total time: {total_time:.1f}s")
            
            return True
                
        except KeyboardInterrupt:
            print("\n⚠️ Cancelled")
            self.is_processing = False
            self.is_playing = False
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            self.cleanup_audio()

def main():
    """Entry point"""
    print("🚀 Starting streaming TTS...")
    
    # Set high priority
    try:
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        print("✅ High priority set")
    except Exception:
        pass
    
    # Run
    reader = OptimizedTTSReader()
    success = reader.run()
    
    if success:
        print("🎊 Streaming completed!")
    else:
        print("❌ Failed")
    
    print("✨ Done!")

if __name__ == "__main__":
    main()