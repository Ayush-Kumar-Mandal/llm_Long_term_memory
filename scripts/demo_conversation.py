"""
demo_conversation.py — demonstrates the full long-form memory system
with a realistic multi-turn conversation showing memory recall across turns.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.injector.injector import process_turn
from src.storage.vector_store import clear_all_memories
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

CONVERSATION = [
    # Early turns — establishing facts
    (1,  "Hi! My name is Ayush and I am from Roorkee."),
    (2,  "I prefer calls after 11 AM only."),
    (3,  "I am vegetarian by the way."),
    (4,  "My manager is Priya and I report to her daily."),
    (5,  "Always respond to me using bullet points."),

    # Middle turns — unrelated queries
    (6,  "What is the capital of Japan?"),
    (7,  "How do I reverse a string in Python?"),
    (8,  "What is the difference between RAM and ROM?"),
    (9,  "Explain what a neural network is."),
    (10, "What is photosynthesis?"),

    # Late turns — testing memory recall
    (50, "Can you schedule a call with me?"),
    (100,"Suggest me a good lunch for today."),
    (200,"What do you know about me?"),
    (500,"Who is my manager?"),
    (937,"Can you call me tomorrow?"),
]


def run_demo():
    console.print(Panel.fit(
        "[bold green]Long-Form Memory System — Live Demo[/bold green]\n"
        "[dim]IITG Hackathon | Episodic + Semantic + Procedural Memory[/dim]",
        border_style="green"
    ))

    # Clear previous memories
    clear_all_memories()
    conversation_history = []

    for turn_num, message in CONVERSATION:
        console.print(f"\n[bold cyan]Turn {turn_num:>4}[/bold cyan] [white]User:[/white] {message}")

        result = process_turn(message, turn_num, conversation_history)

        console.print(f"         [bold yellow]Assistant:[/bold yellow] {result['response']}")

        if result["new_memories_saved"] > 0:
            console.print(f"         [dim green]✓ Saved {result['new_memories_saved']} new memories[/dim green]")

        if turn_num >= 50 and result["active_memories"]:
            active = result["active_memories"]
            if isinstance(active, list):
                turns = [m.get('source_turn') for m in active]
            else:
                turns = []
            console.print(f"         [dim blue]↑ Used memories from turn(s): {turns}[/dim blue]")

    console.print("\n" + "="*60)
    console.print("[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    run_demo()