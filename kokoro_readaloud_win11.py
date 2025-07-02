import os
import re
import sys
import time
import threading
from pathlib import Path
import traceback
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class WindowsTTSReader:
    def __init__(self):
        self.sample_rate = 24000
        self.chunk_size = 1024
        self.output_dir = Path('Output_audio')
        self.kokoro = None
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Performance optimizations
        self.cpu_count = os.cpu_count() or 4  # Default to 4 if detection fails
        self.max_threads = min(4, self.cpu_count)  # Limit threads for stability
        self.chunk_max_words = 50  # Process text in smaller chunks
        
        print(f"🚀 System specs: {self.cpu_count} CPU cores, {self.max_threads} threads for TTS")
        
    def optimize_system_performance(self):
        """Optimize system resources for maximum TTS performance"""
        try:
            # Set high process priority
            current_process = psutil.Process()
            current_process.nice(psutil.HIGH_PRIORITY_CLASS)
            
            # Optimize memory usage
            current_process.memory_percent()  # Trigger memory info update
            
            # Set CPU affinity to use all cores
            cpu_count = os.cpu_count() or 4  # Handle None case
            current_process.cpu_affinity(list(range(cpu_count)))
            
            # Configure environment for optimal ONNX Runtime CPU performance
            os.environ['OMP_NUM_THREADS'] = str(self.cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(self.cpu_count)
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.cpu_count)
            os.environ['BLIS_NUM_THREADS'] = str(self.cpu_count)
            os.environ['ORT_NUM_THREADS'] = str(self.cpu_count)
            
            # Enable all CPU optimizations
            os.environ['ORT_ENABLE_CPU_FP16_OPS'] = '1'
            os.environ['ORT_ENABLE_ALL_OPTIMIZATIONS'] = '1'
            os.environ['ORT_ENABLE_EXTENDED_OPTIMIZATIONS'] = '1'
            
            print(f"✅ System optimized: High priority, {self.cpu_count} threads, full CPU utilization")
            return True
            
        except Exception as e:
            print(f"⚠️ Performance optimization warning: {e}")
            return False
        
    def preprocess_text_for_chunking(self, text):
        """Enhanced text preprocessing with intelligent chunking for parallel processing"""
        if not text or not text.strip():
            return []
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Group sentences into optimal chunks for parallel processing
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Clean up the sentence
            sentence = re.sub(r'\b(Mr|Mrs|Dr|Prof|Inc|Ltd|etc)\.\s+', r'\1. ', sentence)
            sentence = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', sentence)
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            words_in_sentence = len(sentence.split())
            
            # If adding this sentence would exceed chunk limit, start new chunk
            if current_word_count + words_in_sentence > self.chunk_max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = words_in_sentence
            else:
                current_chunk.append(sentence)
                current_word_count += words_in_sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def initialize_audio(self):
        """Initialize PyAudio with enhanced Windows 11 optimizations"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find the best audio device
            default_device = self.pyaudio_instance.get_default_output_device_info()
            print(f"🔊 Audio device: {default_device['name']}")
            
            # Use larger buffer for better performance
            buffer_size = self.chunk_size * 4  # Larger buffer for smoother playback
            
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=buffer_size,
                output_device_index=int(default_device['index'])
            )
            return True
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            return False
    
    def cleanup_audio(self):
        """Clean up audio resources"""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
        except Exception as e:
            print(f"⚠️ Audio cleanup warning: {e}")
    
    def download_optimized_models(self):
        """Download the most performance-optimized Kokoro model files"""
        model_dir = Path("model")
        voice_dir = Path("voice")
        model_dir.mkdir(exist_ok=True)
        voice_dir.mkdir(exist_ok=True)
        
        # Use the quantized int8 model for better performance (smaller and faster)
        model_path = model_dir / "kokoro-v1.0.int8.onnx"
        voices_path = voice_dir / "voices-v1.0.bin"
        
        if model_path.exists() and voices_path.exists():
            print("✅ Optimized models already exist")
            return str(model_path), str(voices_path)
        
        print("📥 Downloading optimized Kokoro model files...")
        print(f"🎯 Target: Quantized INT8 model (88MB) for maximum speed")
        
        try:
            import urllib.request
            
            if not model_path.exists():
                print("⬇️ Downloading optimized INT8 model (~88MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
                    str(model_path)
                )
                print(f"✅ Optimized model downloaded: {model_path}")
            
            if not voices_path.exists():
                print("⬇️ Downloading voices file (~27MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                    str(voices_path)
                )
                print(f"✅ Voices downloaded: {voices_path}")
            
            return str(model_path), str(voices_path)
            
        except Exception as e:
            print(f"❌ Model download failed: {e}")
            return None, None

    def initialize_optimized_tts(self):
        """Initialize TTS with maximum CPU optimization"""
        try:
            print("🎯 Initializing high-performance Kokoro TTS...")
            
            # Optimize system first
            self.optimize_system_performance()
            
            # Download optimized models
            model_path, voices_path = self.download_optimized_models()
            if not model_path or not voices_path:
                return False
            
            print("⚡ Loading quantized INT8 model for maximum speed...")
            
            # Initialize Kokoro with optimized environment already set
            start_time = time.time()
            self.kokoro = Kokoro(model_path, voices_path)
            load_time = time.time() - start_time
            
            print(f"✅ Kokoro TTS ready in {load_time:.1f}s!")
            print(f"🚀 Optimizations: INT8 quantization, {self.cpu_count} CPU threads, high priority")
            print(f"📁 Model: {model_path}")
            print(f"📁 Voices: {voices_path}")
            return True
            
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            return False
    
    def generate_audio_chunk(self, chunk_text, voice='af_heart', speed=1.1):
        """Generate audio for a single text chunk with optimization"""
        try:
            if not chunk_text.strip():
                return None
            
            if self.kokoro is None:
                raise Exception("Kokoro TTS not initialized")
            
            # Generate audio for this chunk
            samples, sample_rate = self.kokoro.create(
                chunk_text,
                voice=voice,
                speed=speed,  # Slightly faster for efficiency
                lang="en-us"
            )
            
            return (samples, sample_rate)
            
        except Exception as e:
            print(f"❌ Chunk generation failed: {e}")
            return None
    
    def generate_audio_parallel(self, text_chunks, voice='af_heart', speed=1.1):
        """Generate audio using parallel processing for maximum speed"""
        print(f"🚀 Processing {len(text_chunks)} chunks with {self.max_threads} threads...")
        
        all_audio_chunks = []
        total_start_time = time.time()
        
        # Process chunks in parallel with progress tracking
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.generate_audio_chunk, chunk, voice, speed): i 
                for i, chunk in enumerate(text_chunks)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        all_audio_chunks.append((chunk_idx, result))
                    completed += 1
                    progress = (completed / len(text_chunks)) * 100
                    print(f"📊 Progress: {progress:.0f}% ({completed}/{len(text_chunks)})")
                except Exception as e:
                    print(f"❌ Chunk {chunk_idx} failed: {e}")
        
        # Sort chunks by original order and combine audio
        all_audio_chunks.sort(key=lambda x: x[0])
        
        if not all_audio_chunks:
            raise Exception("No audio chunks were generated successfully")
        
        # Combine all audio chunks
        combined_samples = []
        sample_rate = all_audio_chunks[0][1][1]  # Use sample rate from first chunk
        
        for _, (samples, _) in all_audio_chunks:
            combined_samples.extend(samples)
        
        combined_samples = np.array(combined_samples, dtype=np.float32)
        
        total_time = time.time() - total_start_time
        audio_duration = len(combined_samples) / sample_rate
        speed_factor = audio_duration / total_time if total_time > 0 else 0
        
        print(f"✅ Parallel processing complete!")
        print(f"   📊 Generated {audio_duration:.1f}s audio in {total_time:.1f}s")
        print(f"   🚀 Speed: {speed_factor:.1f}x real-time with {self.max_threads} threads")
        print(f"   🎯 Total samples: {len(combined_samples)} at {sample_rate}Hz")
        
        return (combined_samples, sample_rate)
    
    def save_and_play_audio(self, audio_data, filename='output.wav'):
        """Save and play audio with optimized streaming"""
        if not audio_data:
            print("❌ No audio data to process")
            return False
        
        try:
            samples, sample_rate = audio_data
            
            # Create output directory
            self.output_dir.mkdir(exist_ok=True)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples / max_val * 0.95
            
            # Save to file
            output_path = self.output_dir / filename
            sf.write(str(output_path), samples, sample_rate)
            print(f"💾 Audio saved: {output_path}")
            
            # Play audio with optimized streaming
            print("🔊 Playing audio...")
            if not self.initialize_audio():
                return False
            
            # Convert to int16 for playback
            samples = np.asarray(samples)
            audio_data_int16 = (samples * 32767).astype(np.int16)
            
            # Stream audio in optimized chunks
            if self.audio_stream is not None:
                chunk_size = self.chunk_size * 2  # Larger chunks for better performance
                for i in range(0, len(audio_data_int16), chunk_size):
                    chunk = audio_data_int16[i:i + chunk_size]
                    self.audio_stream.write(chunk.tobytes())
            
            print("✅ Playback completed!")
            return True
            
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            return False
        finally:
            self.cleanup_audio()
    
    def run(self):
        """Main execution with optimized parallel processing"""
        print("=== Windows 11 Kokoro TTS Reader ===")
        print("⚡ High-Performance Edition with Parallel Processing")
        print(f"🖥️ Intel i7-1255U | 16GB RAM | {self.cpu_count} CPU Cores")
        print()
        
        try:
            # Get text from clipboard
            print("📋 Reading text from clipboard...")
            text = pyperclip.paste()
            if not text or not text.strip():
                print("❌ No text found in clipboard.")
                print("💡 Please copy some text (Ctrl+C) and try again.")
                return False
            
            print(f"✅ Found {len(text)} characters")
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"📝 Preview: '{preview}'")
            print()
            
            # Process text into optimized chunks
            print("🔄 Preprocessing text for parallel processing...")
            text_chunks = self.preprocess_text_for_chunking(text)
            
            if not text_chunks:
                print("❌ Text preprocessing resulted in no valid chunks")
                return False
            
            total_words = sum(len(chunk.split()) for chunk in text_chunks)
            print(f"✅ Text processed: {len(text_chunks)} chunks, {total_words} words")
            print(f"📊 Chunk sizes: {[len(chunk.split()) for chunk in text_chunks[:5]]}{'...' if len(text_chunks) > 5 else ''}")
            
            estimated_time = total_words / 180  # ~180 words per minute
            print(f"⏱️ Estimated speech time: {estimated_time:.1f} minutes")
            print()
            
            # Initialize optimized TTS
            if not self.initialize_optimized_tts():
                return False
            
            # Generate audio with parallel processing
            print("🚀 Starting parallel audio generation...")
            audio_data = self.generate_audio_parallel(text_chunks, voice='af_heart', speed=1.1)
            
            # Save and play
            if self.save_and_play_audio(audio_data):
                print("\n🎉 High-performance TTS completed successfully!")
                print(f"📁 Audio saved in: {self.output_dir}")
                print(f"⚡ Powered by: {self.max_threads}-thread parallel processing")
                return True
            else:
                print("❌ Audio playback failed")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ Operation cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("\n🔍 Error details:")
            traceback.print_exc()
            return False
        finally:
            self.cleanup_audio()

def main():
    """Entry point with maximum performance optimization"""
    print("🚀 Initializing high-performance mode...")
    
    # Set maximum process priority
    try:
        import psutil
        current_process = psutil.Process()
        current_process.nice(psutil.HIGH_PRIORITY_CLASS)
        print("✅ Process priority set to HIGH")
    except (ImportError, Exception) as e:
        print(f"⚠️ Priority optimization unavailable: {e}")
    
    # Initialize and run optimized TTS reader
    reader = WindowsTTSReader()
    success = reader.run()
    
    print()
    if success:
        print("🎊 Mission accomplished! Maximum performance achieved!")
        print("⚡ Your system's full potential has been unleashed!")
    else:
        print("❌ Operation failed. Check error messages above.")
        print("💡 Try: pip install --upgrade kokoro-onnx")
    
    print("✨ Exiting...")

if __name__ == "__main__":
    main()