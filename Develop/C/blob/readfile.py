image = []
with open('conv0.bb','rb') as f:
    pixels = f.read(320 * 160 * 4 * 2)
    print(type(pixels))
    for pixel in pixels:
        image.append(pixel)
print(len(image))
print(max(image))
print(min(image))