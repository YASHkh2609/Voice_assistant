import asyncio
from conversation_manager import ConversationManager

if __name__ == "__main__":
    print("Listening.....")
    manager = ConversationManager()
    try:
        asyncio.run(manager.main())
    except KeyboardInterrupt:
        print("Application interrupted by user.")

