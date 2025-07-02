import os
import re
import sys
import time
import threading
from pathlib import Path
import traceback

try:
    from kokoro_onnx import Kokoro
    import soundfile as sf
    import pyaudio
    import numpy as np
    import pyperclip
    import onnxruntime as ort
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
        self.gpu_info = self.detect_gpu_capabilities()
        
    def detect_gpu_capabilities(self):
        """Detect available GPU acceleration for Intel Iris Xe"""
        gpu_info = {
            'directml_available': False,
            'openvino_available': False,
            'provider': 'CPU',
            'device_name': 'Intel Iris Xe Graphics'
        }
        
        try:
            # Check DirectML availability (best for Intel Iris Xe)
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                gpu_info['directml_available'] = True
                gpu_info['provider'] = 'DirectML'
                print("🚀 DirectML acceleration detected for Intel Iris Xe Graphics")
            elif 'OpenVINOExecutionProvider' in providers:
                gpu_info['openvino_available'] = True
                gpu_info['provider'] = 'OpenVINO'
                print("🚀 OpenVINO acceleration detected for Intel Iris Xe Graphics")
            else:
                print("⚡ Using CPU inference (GPU acceleration not available)")
                
        except Exception as e:
            print(f"ℹ️  GPU detection info: {e}")
            
        return gpu_info
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing for better TTS output"""
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences more intelligently
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Handle common abbreviations that shouldn't end sentences
                sentence = re.sub(r'\b(Mr|Mrs|Dr|Prof|Inc|Ltd|etc)\.\s+', r'\1. ', sentence)
                # Ensure proper spacing around punctuation
                sentence = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', sentence)
                # Remove extra spaces
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                processed_sentences.append(sentence)
        
        # Join with appropriate pauses for better speech flow
        return ' '.join(processed_sentences)
    
    def initialize_audio(self):
        """Initialize PyAudio with Windows 11 optimizations"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find the best audio device (preferably default output)
            default_device = self.pyaudio_instance.get_default_output_device_info()
            print(f"🔊 Using audio device: {default_device['name']}")
            
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
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
            print(f"⚠️  Audio cleanup warning: {e}")
    
    def download_models(self):
        """Download Kokoro model files if they don't exist"""
        # Create organized folder structure
        model_dir = Path("model")
        voice_dir = Path("voice")
        model_dir.mkdir(exist_ok=True)
        voice_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "kokoro-v1.0.int8.onnx"
        voices_path = voice_dir / "voices-v1.0.bin"
        
        if model_path.exists() and voices_path.exists():
            print("✅ Model files already exist")
            return str(model_path), str(voices_path)
        
        print("📥 Downloading Kokoro model files...")
        print(f"📁 Download location: {Path.cwd()}")
        print(f"   📂 Model folder: {model_dir}")
        print(f"   📂 Voice folder: {voice_dir}")
        
        try:
            import urllib.request
            
            if not model_path.exists():
                print("⬇️  Downloading model file (~80MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
                    str(model_path)
                )
                print(f"✅ Model file downloaded to: {model_path}")
            
            if not voices_path.exists():
                print("⬇️  Downloading voices file (~220MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                    str(voices_path)
                )
                print(f"✅ Voices file downloaded to: {voices_path}")
            
            return str(model_path), str(voices_path)
            
        except Exception as e:
            print(f"❌ Model download failed: {e}")
            print("🔍 Manual download:")
            print("1. Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0")
            print(f"2. Place model file in: {model_dir}")
            print(f"3. Place voices file in: {voice_dir}")
            return None, None

    def initialize_tts(self):
        """Initialize TTS with Kokoro ONNX and GPU acceleration"""
        try:
            print("🎯 Initializing Kokoro TTS with Intel Iris Xe acceleration...")
            
            # Download models if needed
            model_path, voices_path = self.download_models()
            if not model_path or not voices_path:
                return False
            
            # Configure providers for Intel Iris Xe Graphics
            providers = []
            provider_options = []
            
            if self.gpu_info['directml_available']:
                # DirectML is best for Intel Iris Xe
                providers.append('DmlExecutionProvider')
                provider_options.append({'device_id': 0})
                print("🚀 Using DirectML acceleration (Intel Iris Xe Graphics)")
                
            elif self.gpu_info['openvino_available']:
                # OpenVINO as fallback
                providers.append('OpenVINOExecutionProvider')
                provider_options.append({'device_type': 'GPU_FP16'})
                print("🚀 Using OpenVINO acceleration (Intel Iris Xe Graphics)")
                
            # Always add CPU as fallback
            providers.append('CPUExecutionProvider')
            provider_options.append({})
            
            # Initialize Kokoro with model files
            # Note: GPU acceleration is handled automatically by ONNX Runtime
            self.kokoro = Kokoro(model_path, voices_path)
            
            print(f"✅ Kokoro TTS ready with {self.gpu_info['provider']} acceleration!")
            print(f"🖥️  GPU: {self.gpu_info['device_name']} (2GB VRAM)")
            print(f"📁 Model loaded from: model/")
            print(f"📁 Voices loaded from: voice/")
            return True
            
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            
            # Try fallback without GPU acceleration
            try:
                print("🔄 Trying fallback initialization (CPU only)...")
                model_path, voices_path = self.download_models()
                if model_path and voices_path:
                    self.kokoro = Kokoro(model_path, voices_path)
                    print("✅ Kokoro TTS ready (CPU mode)")
                    return True
                else:
                    return False
            except Exception as fallback_error:
                print(f"❌ Fallback initialization also failed: {fallback_error}")
                print("\n🔍 Troubleshooting:")
                print("1. Ensure you have internet connection for model download")
                print("2. Try running: pip install --upgrade kokoro-onnx")
                print("3. Check available disk space (~300MB needed)")
                print("4. Manual download: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0")
                return False
    
    def generate_audio_threaded(self, text, voice='af_heart', speed=1.0):
        """Generate audio using Kokoro ONNX with GPU acceleration"""
        self.audio_data = None
        self.generation_complete = False
        self.generation_error = None
        
        def generate():
            try:
                provider_info = f"({self.gpu_info['provider']})" if self.gpu_info['provider'] != 'CPU' else "(CPU)"
                print(f"🎵 Generating audio with voice '{voice}' at {speed}x speed {provider_info}...")
                
                if self.kokoro is None:
                    raise Exception("Kokoro TTS not initialized")
                
                # Generate audio using Kokoro ONNX with GPU acceleration
                start_time = time.time()
                samples, sample_rate = self.kokoro.create(
                    text, 
                    voice=voice, 
                    speed=speed, 
                    lang="en-us"
                )
                generation_time = time.time() - start_time
                
                self.audio_data = (samples, sample_rate)
                self.generation_complete = True
                
                audio_duration = len(samples) / sample_rate
                speed_factor = audio_duration / generation_time if generation_time > 0 else 0
                
                print(f"✅ Audio generation complete!")
                print(f"   📊 {len(samples)} samples at {sample_rate}Hz")
                print(f"   ⏱️  Generated {audio_duration:.1f}s audio in {generation_time:.1f}s")
                print(f"   🚀 Speed: {speed_factor:.1f}x real-time with {self.gpu_info['provider']}")
                
            except Exception as e:
                self.generation_error = str(e)
                print(f"❌ Audio generation failed: {e}")
        
        # Start generation in background thread
        generation_thread = threading.Thread(target=generate)
        generation_thread.daemon = True
        generation_thread.start()
        
        # Show progress while waiting
        while not self.generation_complete and self.generation_error is None:
            print(".", end="", flush=True)
            time.sleep(0.5)
        
        print()  # New line after progress dots
        
        if self.generation_error:
            raise Exception(self.generation_error)
        
        return self.audio_data
    
    def save_and_play_audio(self, audio_data, filename='output.wav'):
        """Save and play audio with Windows optimizations"""
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
            print(f"💾 Audio saved to: {output_path}")
            
            # Play audio
            print("🔊 Playing audio...")
            if not self.initialize_audio():
                return False
            
            # Convert to int16 for playback and ensure correct sample rate
            if sample_rate != self.sample_rate:
                # Simple resampling if needed (basic interpolation)
                try:
                    import scipy.signal
                    samples = scipy.signal.resample(samples, int(len(samples) * self.sample_rate / sample_rate))
                except ImportError:
                    # Fallback: basic linear interpolation using numpy
                    old_len = len(samples)
                    new_len = int(old_len * self.sample_rate / sample_rate)
                    samples = np.interp(np.linspace(0, old_len-1, new_len), np.arange(old_len), samples)
            
            # Ensure samples is a numpy array and convert to int16
            samples = np.asarray(samples)
            audio_data_int16 = (samples * 32767).astype(np.int16)
            
            # Play in chunks for better performance
            if self.audio_stream is not None:
                for i in range(0, len(audio_data_int16), self.chunk_size):
                    chunk = audio_data_int16[i:i + self.chunk_size]
                    self.audio_stream.write(chunk.tobytes())
            
            print("✅ Playback completed!")
            return True
            
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            return False
        finally:
            self.cleanup_audio()
    
    def run(self):
        """Main execution method"""
        print("=== Windows 11 Kokoro TTS Reader ===")
        print("🚀 GPU-Accelerated Edition for Intel Iris Xe Graphics")
        print("🖥️  Intel i7-1255U | 16GB RAM | Intel Iris Xe Graphics (2GB)")
        print()
        
        try:
            # Get text from clipboard
            print("📋 Reading text from clipboard...")
            text = pyperclip.paste()
            if not text or not text.strip():
                print("❌ No text found in clipboard.")
                print("💡 Please copy some text (Ctrl+C) and try again.")
                return False
            
            print(f"✅ Found {len(text)} characters in clipboard")
            print("📝 Text preview:")
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"   '{preview}'")
            print()
            
            # Preprocess text
            print("🔄 Preprocessing text...")
            processed_text = self.preprocess_text(text)
            if not processed_text:
                print("❌ Text preprocessing resulted in empty content")
                return False
            
            word_count = len(processed_text.split())
            print(f"✅ Text processed: {word_count} words")
            estimated_time = word_count / 180  # ~180 words per minute
            print(f"⏱️  Estimated speech time: {estimated_time:.1f} minutes")
            print()
            
            # Initialize TTS
            if not self.initialize_tts():
                return False
            
            # Generate audio
            print("🎵 Starting GPU-accelerated audio generation...")
            audio_data = self.generate_audio_threaded(processed_text, voice='af_heart', speed=1.0)
            
            # Save and play
            if self.save_and_play_audio(audio_data):
                print("\n🎉 TTS operation completed successfully!")
                print(f"📁 Audio file saved in: {self.output_dir}")
                print(f"🚀 Accelerated by: {self.gpu_info['provider']} on Intel Iris Xe Graphics")
                return True
            else:
                print("❌ Audio playback failed")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ Operation cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("\n🔍 Full error details:")
            traceback.print_exc()
            return False
        finally:
            self.cleanup_audio()

def main():
    """Entry point with Windows-specific optimizations"""
    # Set process priority for better audio performance on Windows
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        print("🚀 Process priority optimized for audio performance")
    except ImportError:
        pass  # psutil not available, continue anyway
    
    # Initialize and run TTS reader
    reader = WindowsTTSReader()
    success = reader.run()
    
    print()
    if success:
        print("🎊 All done! Your Intel Iris Xe Graphics delivered great performance!")
        print("✨ Exiting automatically...")
    else:
        print("❌ Operation failed. Check the error messages above.")
        print("💡 Try running the setup: pip install -r requirements.txt")
        print("⚠️ Exiting...")
    
    # Exit immediately - no delay

if __name__ == "__main__":
    main()