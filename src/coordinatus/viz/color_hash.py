import hashlib

def rgb2hsl(r:float, g: float, b: float):
    mx = max((r, g, b))
    mn = min((r, g, b))
    h = s = lum = (mx + mn) / 2
    if mx == mn:
        h = s = 0
    else:
        d = mx - mn
        s = d / (1 - abs(2 * lum - 1))
        if mx == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif mx == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6
    return int(h * 360), int(s * 100), int(lum * 100)

def hsl2rgb(hue: float, sat: float, lightness: float):
    hue /= 360
    sat /= 100
    lum = lightness / 100
    if sat == 0:
        red = grn = blu = int(lum * 255)
    else:
        def hue2rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        q = lum * (1 + sat) if lum < 0.5 else lum + sat - lum * sat
        p = 2 * lum - q
        red = int(hue2rgb(p, q, hue + 1/3) * 255)
        grn = int(hue2rgb(p, q, hue) * 255)
        blu = int(hue2rgb(p, q, hue - 1/3) * 255)
    return red, grn, blu

def to_hex(r: float, g: float, b: float) -> str:
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

def generate(seed: str):
    def scale(x, min_val, max_val):
        return min_val + x*(max_val - min_val)
    hash_value = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    u = ((hash_value & 0xFFF0000) >> 16)/16**3
    v = ((hash_value & 0x000FF00) >> 8)/256
    w = (hash_value &  0x00000FF)/256

    hue = u * 360
    sat = scale(v, 50, 100)  # Saturation between 50% and 100%
    lum = scale(w, 40, 60)  # Lightness between 40% and 60%

    red, grn, blu = hsl2rgb(hue, sat, lum)
    return to_hex(red, grn, blu)