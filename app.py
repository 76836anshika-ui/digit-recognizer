"""Draw a digit with your mouse and get a live prediction.

Usage:
    python app.py            (needs a trained digit_cnn.pt; Tkinter ships with Python)
"""
import tkinter as tk

from PIL import Image, ImageDraw

from predict import load_model, predict

CANVAS = 280  # 10x the MNIST size
BRUSH = 11


class App:
    def __init__(self, root):
        self.model, self.device = load_model()
        root.title("Handwritten Digit Recognizer")

        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black",
                                cursor="crosshair", highlightthickness=1)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.recognize())

        self.img = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.img)

        self.result = tk.Label(root, text="Draw a digit (0-9)", font=("Helvetica", 18))
        self.result.grid(row=1, column=0, columnspan=2)
        self.bars = tk.Label(root, text="", font=("Courier", 10), justify="left")
        self.bars.grid(row=2, column=0, columnspan=2, pady=5)
        tk.Button(root, text="Clear", command=self.clear).grid(row=3, column=0, columnspan=2, pady=8)

    def paint(self, e):
        r = BRUSH
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white", outline="white")
        self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def recognize(self):
        try:
            digit, probs = predict(self.model, self.device, self.img)
        except ValueError:
            return
        self.result.config(text=f"Prediction: {digit}  ({probs[digit]:.0%})")
        top = sorted(range(10), key=lambda d: -probs[d])[:3]
        self.bars.config(text="\n".join(f"{d}: {'█' * int(probs[d] * 30):<30} {probs[d]:.1%}" for d in top))

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS, CANVAS], fill=0)
        self.result.config(text="Draw a digit (0-9)")
        self.bars.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
