#!/usr/bin/env python3
"""
Setup script for the RAG Chatbot project
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a system command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    system = platform.system().lower()
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Get activation command based on OS
    if system == "windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"📝 To activate the virtual environment, run:")
    if system == "windows":
        print(f"   {activate_cmd}")
    else:
        print(f"   {activate_cmd}")
    
    return pip_cmd

def install_dependencies(pip_cmd):
    """Install required dependencies"""
    return run_command(f"{pip_cmd} install -r requirements.txt", 
                      "Installing dependencies")

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            run_command("cp .env.example .env", "Creating .env file")
            print("📝 Please edit .env file and add your OpenAI API key")
        else:
            print("⚠️ .env.example not found, please create .env manually")
    else:
        print("✅ .env file already exists")

def run_tests():
    """Run basic tests to verify setup"""
    print("🧪 Running basic tests...")
    try:
        # Test imports
        import langchain
        import streamlit
        import openai
        import faiss
        import wikipedia
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up RAG Chatbot...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    pip_cmd = create_virtual_environment()
    if not pip_cmd:
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(pip_cmd):
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    
    system = platform.system().lower()
    if system == "windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("2. Edit .env file and add your OpenAI API key")
    print("3. Run the Streamlit app:")
    print("   streamlit run app.py")
    print("4. Or run the CLI version:")
    print("   python cli_rag.py")
    
    print("\n🔗 Useful links:")
    print("- OpenAI API Keys: https://platform.openai.com/api-keys")
    print("- Documentation: README.md")
    print("- Issues: Report any problems in the GitHub repository")

if __name__ == "__main__":
    main()