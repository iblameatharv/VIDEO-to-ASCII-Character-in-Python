import cv2
import numpy as np
import argparse
from typing import Tuple, Dict

# Extended ASCII characters for better detail
ASCII_CHARS = "@%#*+=-:. "

# Enhanced color map for ASCII characters
COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "@": (255, 255, 255),
    "%": (200, 200, 200),
    "#": (150, 150, 150),
    "*": (100, 100, 100),
    "+": (70, 70, 70),
    "=": (50, 50, 50),
    "-": (30, 30, 30),
    ":": (20, 20, 20),
    ".": (10, 10, 10),
    " ": (0, 0, 0)
}

def resize_image(image: np.ndarray, new_width: int = 100) -> np.ndarray:
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.55)
    return cv2.resize(image, (new_width, new_height))

def adjust_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    # Adjust the contrast of the image
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def grayify(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def pixels_to_ascii(image: np.ndarray) -> str:
    # Lookup table for ASCII conversion
    lut = np.array([ASCII_CHARS[min(i // 25, len(ASCII_CHARS) - 1)] for i in range(256)])
    return ''.join(lut[image.flatten()])

def convert_to_ascii(image: np.ndarray) -> str:
    image = resize_image(image)
    image = grayify(image)
    image = adjust_contrast(image, alpha=2.0, beta=-20)
    ascii_str = pixels_to_ascii(image)
    img_width = image.shape[1]
    return '\n'.join([ascii_str[i:i + img_width] for i in range(0, len(ascii_str), img_width)])

def get_ansi_code(color: Tuple[int, int, int]) -> str:
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m"

def colorize_ascii(ascii_str: str) -> str:
    return ''.join(get_ansi_code(COLOR_MAP.get(char, (255, 255, 255))) + char for char in ascii_str) + "\033[0m"

def process_video(video_path: str, output_width: int, save_output: bool = False) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = int(1000 / fps)  # Convert frame time to milliseconds

    if save_output:
        output_file = open("ascii_output.txt", "w")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            try:
                ascii_image = convert_to_ascii(frame)
                colored_ascii = colorize_ascii(ascii_image)

                if save_output:
                    output_file.write(ascii_image + "\n\n")
                else:
                    print("\033c", end="")  # Clear the terminal
                    print(colored_ascii, end="")

                cv2.waitKey(frame_time)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("Video playback interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_output:
            output_file.close()

def main():
    parser = argparse.ArgumentParser(description="Convert video to ASCII art")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--width", type=int, default=100, help="Width of the ASCII output")
    parser.add_argument("--save", action="store_true", help="Save ASCII output to a text file")
    args = parser.parse_args()

    try:
        process_video(args.video_path, args.width, args.save)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
