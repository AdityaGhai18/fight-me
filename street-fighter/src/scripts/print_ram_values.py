import retro
import numpy as np
import time
import os

def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED, render_mode="human")
    env.reset()
    
    # RAM range: 0x900000-0x92FFFF (196608 bytes)
    RAM_START = 0x900000
    RAM_END = 0x92FFFF
    RAM_SIZE = RAM_END - RAM_START + 1  # 196608
    WINDOW_SIZE = 256  # Number of bytes to display at once
    window_start = 0   # Start index in the RAM array
    
    print("Use Ctrl+C to stop. Showing RAM window for 0x900000-0x92FFFF. Character will only move RIGHT.")
    time.sleep(2)
    
    prev_ram = env.get_ram()
    right_action = np.zeros(12, dtype=np.uint8)
    right_action[7] = 1  # RIGHT button
    try:
        while True:
            ram = env.get_ram()
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"RAM 0x900000-0x92FFFF (showing {WINDOW_SIZE} bytes, offset {window_start}):")
            print("Offset | " + " ".join(f"{i:02X}" for i in range(16)))
            print("-" * 80)
            for row in range(window_start, window_start + WINDOW_SIZE, 16):
                values = ram[row:row+16]
                prev_values = prev_ram[row:row+16]
                row_str = f"0x{RAM_START+row:06X} | "
                for curr, prev in zip(values, prev_values):
                    if curr != prev:
                        row_str += f"\033[91m{curr:02X}\033[0m "
                    else:
                        row_str += f"{curr:02X} "
                print(row_str)
            print("\nPress 'n' then Enter to view next window, 'p' for previous, or just Enter to refresh.")
            prev_ram = ram.copy()
            # Always move right
            env.step(right_action)
            # User input for paging
            import select, sys
            print("Waiting 0.2s for input...")
            i, o, e = select.select([sys.stdin], [], [], 0.2)
            if i:
                cmd = sys.stdin.readline().strip()
                if cmd == 'n':
                    window_start = min(window_start + WINDOW_SIZE, RAM_SIZE - WINDOW_SIZE)
                elif cmd == 'p':
                    window_start = max(window_start - WINDOW_SIZE, 0)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
