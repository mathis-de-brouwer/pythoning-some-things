#!/usr/bin/env python3
import subprocess
import sys
import os
import platform

def main():
    print("🚀 Setting up environment for Encrypted Video Streaming System")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
        print("❌ Error: Python 3.6 or higher is required")
        sys.exit(1)
    
    print(f"✅ Using Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        sys.exit(1)
    
    # System-specific dependencies
    system = platform.system()
    if system == "Linux":
        distro = platform.linux_distribution()[0].lower() if hasattr(platform, "linux_distribution") else ""
        
        if "ubuntu" in distro or "debian" in distro:
            print("\n🐧 Detected Ubuntu/Debian - checking additional dependencies")
            try:
                subprocess.check_call(["apt-get", "--version"], stdout=subprocess.DEVNULL)
                print("Installing libgl1-mesa-glx (required for OpenCV)")
                subprocess.call(["sudo", "apt-get", "update", "-y"])
                subprocess.call(["sudo", "apt-get", "install", "-y", "libgl1-mesa-glx"])
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️ Warning: Could not check/install system dependencies")
                print("   You may need to manually install: libgl1-mesa-glx")
    
    # Create necessary directories
    print("\n📁 Creating necessary directories...")
    os.makedirs("extracted_frames", exist_ok=True)
    os.makedirs(os.path.join("testing some things", "received_frames"), exist_ok=True)
    
    print("\n✅ Setup complete! You can now run the following commands:")
    print("\n1. Start the signaling server:")
    print("   python signalingServer.py")
    print("\n2. Start the receiver:")
    print("   python testing\\ some\\ things/recievertest1.py --save-frames")
    print("\n3. Start the sender (webcam):")
    print("   python testing\\ some\\ things/sendertest1.py --source 0")
    print("\n   Or with a video file:")
    print("   python testing\\ some\\ things/sendertest1.py --source /path/to/video.mp4")

if __name__ == "__main__":
    main()