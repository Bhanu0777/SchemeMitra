"""
🏛️ SchemeMitra - Quick Start Guide for Windows Users

This script helps you set up SchemeMitra quickly on Windows.
It will install dependencies and guide you through Azure setup.
"""

import os
import sys
import subprocess
import json

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_success(text):
    """Print success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print error message."""
    print(f"❌ {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")

def check_python_version():
    """Check if Python 3.9+ is installed."""
    print_header("CHECKING PYTHON INSTALLATION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print_success(f"Python {version.major}.{version.minor} detected")
        return True
    else:
        print_error(f"Python 3.9+ required. You have {version.major}.{version.minor}")
        return False

def install_dependencies():
    """Install required packages from requirements.txt."""
    print_header("INSTALLING DEPENDENCIES")
    
    if not os.path.exists('requirements.txt'):
        print_error("requirements.txt not found!")
        return False
    
    try:
        print_info("Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print_success("All dependencies installed!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def create_venv():
    """Create a Python virtual environment."""
    print_header("CREATING VIRTUAL ENVIRONMENT (Optional but Recommended)")
    
    venv_name = "venv"
    
    if os.path.exists(venv_name):
        print_info(f"Virtual environment '{venv_name}' already exists")
        return True
    
    try:
        print_info(f"Creating virtual environment '{venv_name}'...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print_success(f"Virtual environment created!")
        
        # Print activation instructions
        print_info("To activate the virtual environment, run:")
        print(f"    {venv_name}\\Scripts\\activate")
        
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def setup_env_file():
    """Guide user through .env file setup."""
    print_header("SETTING UP AZURE CREDENTIALS (.env FILE)")
    
    if os.path.exists('.env'):
        print_info(".env file already exists")
        return True
    
    print_info("Copying .env.example to .env...")
    
    try:
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as f:
                content = f.read()
            
            with open('.env', 'w') as f:
                f.write(content)
            
            print_success(".env file created!")
            print_info("Please edit .env with your Azure credentials:")
            print("    - AZURE_OPENAI_API_KEY")
            print("    - AZURE_OPENAI_ENDPOINT")
            print("    - AZURE_OPENAI_DEPLOYMENT_NAME")
            print("    - AZURE_TEXTANALYTICS_KEY")
            print("    - AZURE_TEXTANALYTICS_ENDPOINT")
            
            return True
        else:
            print_error(".env.example not found")
            return False
    
    except Exception as e:
        print_error(f"Failed to create .env: {e}")
        return False

def verify_data_files():
    """Verify that data files exist."""
    print_header("VERIFYING DATA FILES")
    
    files_to_check = {
        'schemes.json': 'Scheme database',
        'app.py': 'Main application',
        'requirements.txt': 'Dependencies'
    }
    
    all_exist = True
    
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            print_success(f"{filename}: {description}")
        else:
            print_error(f"{filename}: NOT FOUND!")
            all_exist = False
    
    return all_exist

def verify_schemes_data():
    """Verify schemes.json has valid data."""
    print_header("VERIFYING SCHEMES DATA")
    
    try:
        with open('schemes.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        schemes = data.get('schemes', [])
        print_success(f"schemes.json contains {len(schemes)} schemes")
        
        if schemes:
            print_info("Sample schemes:")
            for scheme in schemes[:3]:
                print(f"  • {scheme['name']} ({scheme['ministry']})")
        
        return True
    
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in schemes.json: {e}")
        return False
    except Exception as e:
        print_error(f"Error reading schemes.json: {e}")
        return False

def run_app():
    """Run the Streamlit application."""
    print_header("STARTING APPLICATION")
    
    print_info("Starting SchemeMitra with Streamlit...")
    print("(Make sure you've configured your .env file with Azure credentials)")
    print("\nApp will open at: http://localhost:8501\n")
    
    try:
        subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print_error(f"Failed to run app: {e}")

def main():
    """Main setup routine."""
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🏛️  SchemeMitra - AI Government Scheme Finder".ljust(68) + "║")
    print("║" + "  Quick Start Setup (Windows)".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Run setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Verify Data Files", verify_data_files),
        ("Verify Schemes Data", verify_schemes_data),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print_error(f"Setup failed at: {step_name}")
            return False
    
    print_header("SETUP SUCCESSFUL!")
    
    print("""
✅ All checks passed!

NEXT STEPS:
───────────
1. Configure Azure Credentials:
   - Edit the .env file with your Azure credentials
   - Instructions are in .env file and README.md
   - Get credentials from Azure Portal

2. Install Dependencies:
   - Run: pip install -r requirements.txt

3. Run the Application:
   - Run: streamlit run app.py
   - Or run the setup script again and select 'Run App'

4. Open in Browser:
   - Navigate to http://localhost:8501
   - Start discovering government schemes!

HELPFUL LINKS:
──────────────
📖 README.md - Full setup guide and documentation
🔑 .env.example - Azure credentials template
📋 schemes.json - Real government schemes database

TROUBLESHOOTING:
────────────────
❌ "ModuleNotFoundError" → Run: pip install -r requirements.txt
❌ "Azure error" → Check .env file has correct credentials
❌ "Port already in use" → Run: streamlit run app.py --server.port 8502

Enjoy SchemeMitra! 🎉
    """)
    
    # Ask user what to do next
    print("\nWhat would you like to do?")
    print("1. Run the application now (requires .env setup)")
    print("2. Exit setup")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        run_app()
    else:
        print("\nSetup complete. Run 'streamlit run app.py' when ready!")

if __name__ == "__main__":
    main()
