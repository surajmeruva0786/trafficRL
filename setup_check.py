"""
Quick setup and test script.

This script helps verify that the project is set up correctly
and all dependencies are installed.
"""

import sys
import os


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking Python dependencies...")
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'traci': 'TraCI (SUMO)',
        'sumolib': 'SUMOlib'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (NOT INSTALLED)")
            all_ok = False
    
    return all_ok


def check_sumo():
    """Check if SUMO is installed and accessible."""
    print("\nChecking SUMO installation...")
    
    import subprocess
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"✓ SUMO found: {version}")
            return True
        else:
            print("✗ SUMO command failed")
            return False
    except FileNotFoundError:
        print("✗ SUMO not found in PATH")
        print("  Please install SUMO and add it to your PATH")
        print("  Download from: https://www.eclipse.org/sumo/")
        return False
    except Exception as e:
        print(f"✗ Error checking SUMO: {e}")
        return False


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'traffic_rl',
        'traffic_rl/env',
        'traffic_rl/dqn',
        'traffic_rl/config',
        'traffic_rl/utils',
        'traffic_rl/sumo',
        'scripts'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'traffic_rl/config/config.yaml',
        'traffic_rl/sumo/network.net.xml',
        'scripts/train_dqn.py',
        'scripts/evaluate_dqn.py'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (MISSING)")
            all_ok = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (MISSING)")
            all_ok = False
    
    return all_ok


def test_imports():
    """Test if project modules can be imported."""
    print("\nTesting project imports...")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    modules = [
        'traffic_rl.env.sumo_env',
        'traffic_rl.env.route_generator',
        'traffic_rl.dqn.agent',
        'traffic_rl.dqn.network',
        'traffic_rl.dqn.replay_buffer',
        'traffic_rl.utils.metrics',
        'traffic_rl.utils.plotting',
        'traffic_rl.utils.logging_utils'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module} ({e})")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("=" * 80)
    print("Traffic Signal RL - Setup Verification")
    print("=" * 80)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("SUMO", check_sumo),
        ("Project Structure", check_project_structure),
        ("Module Imports", test_imports)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. python scripts/train_dqn.py")
        print("  2. python scripts/evaluate_dqn.py")
        print("  3. python scripts/run_fixed_time_baseline.py")
        print("  4. python scripts/compare_results.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Install SUMO: https://www.eclipse.org/sumo/")
        print("  - Add SUMO to PATH")
    
    print()


if __name__ == "__main__":
    main()
