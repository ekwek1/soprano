"""
Soprano TTS Server
Menu-driven interface to launch different server options.
"""
import sys
import os
import subprocess


def display_menu() -> None:
    """Display the main menu options"""
    print("\n" + "="*60)
    print("           SOPRANO TTS SERVER MENU")
    print("="*60)
    print("Select an option:")
    print()
    print("1. Start API Server")
    print("   OpenAI-compatible API for workflow integration")
    print("   Accessible at http://localhost:8000/v1/audio/speech")
    print()
    print("2. Test API Server")
    print("   Test client for the OpenAI-compatible API")
    print("   Requires API server to be running")
    print()
    print("3. Start WebSocket Server")
    print("   Real-time audio streaming for interactive applications")
    print("   Available at ws://localhost:8001/ws/tts")
    print()
    print("4. Test WebSocket Server")
    print("   Test client for real-time audio streaming")
    print("   Requires WebSocket server to be running")
    print()
    print("5. Start WebUI")
    print("   Gradio web interface for Soprano TTS")
    print("   Opens browser with interactive UI")
    print()
    print("6. Start CLI")
    print("   Command-line interface for Soprano TTS")
    print("   Interactive menu for text synthesis")
    print()
    print("7. Exit")
    print("="*60)


def get_user_choice() -> str:
    """Get and validate user choice from the menu"""
    while True:
        try:
            choice = input("Enter your choice (1-7): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7']:
                return choice
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, or 7.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nSoprano TTS Server menu interrupted. Goodbye!")
            sys.exit(0)


def main_menu() -> None:
    """Display the main menu and handle user selection"""
    # Get the root directory (where this script is located)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    while True:
        display_menu()

        choice = get_user_choice()

        if choice == '1':
            # Open new cmdline in root, cd to soprano/server, run python api.py
            cmd = f'start cmd /k "cd /d {root_dir} && cd soprano && cd server && python api.py"'
            subprocess.run(cmd, shell=True)
            print("API server started in new terminal. This menu will now close.")
            sys.exit(0)
        elif choice == '2':
            # Open new cmdline in root, cd to soprano/server, run python api.py and test_api.py in separate terminals
            cmd1 = f'start cmd /k "cd /d {root_dir} && cd soprano && cd server && python api.py"'
            cmd2 = f'start cmd /k "cd /d {root_dir} && cd soprano && cd server && ping 127.0.0.1 -n 11 > nul && python test_api.py"'
            subprocess.run(cmd1, shell=True)
            subprocess.run(cmd2, shell=True)
            print("API server and test client started in new terminals. This menu will now close.")
            sys.exit(0)
        elif choice == '3':
            # Open new cmdline in root, cd to soprano/server, run python websocket.py
            cmd = f'start cmd /k "cd /d {root_dir} && cd soprano && cd server && python websocket.py"'
            subprocess.run(cmd, shell=True)
            print("WebSocket server started in new terminal. This menu will now close.")
            sys.exit(0)
        elif choice == '4':
            # Open new cmdline in root, cd to soprano/server, run python websocket.py and test_websocket.py in separate terminals
            cmd1 = f'start cmd /k "cd /d {root_dir} && cd soprano && cd server && python websocket.py"'
            cmd2 = f'start cmd /k "cd /d {root_dir} && cd soprano && cd server && ping 127.0.0.1 -n 11 > nul && python test_websocket.py"'
            subprocess.run(cmd1, shell=True)
            subprocess.run(cmd2, shell=True)
            print("WebSocket server and test client started in new terminals. This menu will now close.")
            sys.exit(0)
        elif choice == '5':
            # Open new cmdline in root, cd to soprano, run python webui.py
            cmd = f'start cmd /k "cd /d {root_dir} && cd soprano && python webui.py"'
            subprocess.run(cmd, shell=True)
            print("WebUI started in new terminal. This menu will now close.")
            sys.exit(0)
        elif choice == '6':
            # Open new cmdline in root, cd to soprano, run python soprano_cli.py
            cmd = f'start cmd /k "cd /d {root_dir} && cd soprano && python soprano_cli.py"'
            subprocess.run(cmd, shell=True)
            print("CLI started in new terminal. This menu will now close.")
            sys.exit(0)
        elif choice == '7':
            print("Thank you for using Soprano TTS. Goodbye!")
            sys.exit(0)


def main() -> None:
    """
    Main entry point for the server module.
    Initializes device detection and starts the main menu.
    """
    try:
        # Check available device
        try:
            import torch
            device_info = f"Available device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}"
            print(device_info)
        except ImportError:
            print("Available device: CPU (PyTorch not available)")

        # Start the main menu
        main_menu()
    except KeyboardInterrupt:
        print("\n\nSoprano TTS Server interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()