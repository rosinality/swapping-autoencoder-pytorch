import argparse

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision import utils


from model import Encoder, Generator


def render(text, size, font, background=(0, 0, 0), foreground=(85, 255, 85)):
    total_height = 0
    max_width = 0

    for line in text.split("\n"):
        text_width, text_height = font.getsize(line)
        max_width = max(max_width, text_width)
        total_height += text_height

    width, height = size
    start_w = max((width - max_width) // 2, 0)
    start_h = max((height - total_height) // 2, 0)

    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    draw.text((start_w, start_h), text, font=font, fill=foreground)

    return image


def pil_to_tensor(pil_img):
    return (
        torch.from_numpy(np.array(pil_img))
        .to(torch.float32)
        .div(255)
        .add(-0.5)
        .mul(2)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="generated.png")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("files", type=str, nargs="+")

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]
    imgsize = ckpt_args.size

    enc = Encoder(ckpt_args.channel)
    gen = Generator(ckpt_args.channel)
    enc.load_state_dict(ckpt["e_ema"])
    gen.load_state_dict(ckpt["g_ema"])
    enc.eval()
    gen.eval()

    imgs = []

    for imgpath in args.files[: len(args.files) // 2 * 2]:
        img = Image.open(imgpath).convert("RGB").resize((imgsize, imgsize))
        img_a = (
            torch.from_numpy(np.array(img))
            .to(torch.float32)
            .div(255)
            .add_(-0.5)
            .mul_(2)
            .permute(2, 0, 1)
        )
        imgs.append(img_a)

    imgs = torch.stack(imgs, 0)
    img1, img2 = imgs.chunk(2, dim=0)

    with torch.no_grad():
        struct1, texture1 = enc(img1)
        struct2, texture2 = enc(img2)

        out1 = gen(struct1, texture1)
        out2 = gen(struct2, texture2)
        out12 = gen(struct1, texture2)
        out21 = gen(struct2, texture1)

    font = ImageFont.truetype(
        "/root/works/sandbox/swapping-autoencoder/Px437_IBM_VGA_8x16.ttf", 16
    )

    guide1 = render('original\nfirst half of batch → "A"', (256, 256), font)
    guide2 = render('reconstruction of "A"', (256, 256), font)
    guide3 = render('original\nsecond half of batch → "B"', (256, 256), font)
    guide4 = render('reconstruction of "B"', (256, 256), font)
    guide5 = render(
        'swapped image\nstructure of "A"\n+\ntexture of "B"', (256, 256), font
    )
    guide6 = render(
        'swapped image\nstructure of "B"\n+\ntexture of "A"', (256, 256), font
    )

    imgsets = [
        pil_to_tensor(guide1),
        img1,
        pil_to_tensor(guide2),
        out1,
        pil_to_tensor(guide3),
        img2,
        pil_to_tensor(guide4),
        out2,
        pil_to_tensor(guide5),
        out12,
        pil_to_tensor(guide6),
        out21,
    ]
    imgsets = torch.cat(imgsets, 0)
    grid = utils.save_image(
        imgsets, args.out, nrow=out1.shape[0] + 1, normalize=True, range=(-1, 1)
    )

