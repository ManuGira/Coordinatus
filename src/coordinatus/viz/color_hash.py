
def rgb2hsl(r:float, g: float, b: float):
    mx = max((r, g, b))
    mn = min((r, g, b))
    h = s = l = (mx + mn) / 2
    if mx == mn:
        h = s = 0
    else:
        d = mx - mn
        s = d / (1 - abs(2 * l - 1))
        if mx == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif mx == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6
    return int(h * 360), int(s * 100), int(l * 100)

def hsl2rgb(h: float, s: float, l: float):
    h /= 360
    s /= 100
    l /= 100
    if s == 0:
        r = g = b = int(l * 255)
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

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = int(hue2rgb(p, q, h + 1/3) * 255)
        g = int(hue2rgb(p, q, h) * 255)
        b = int(hue2rgb(p, q, h - 1/3) * 255)
    return r, g, b

def to_hex(r: float, g: float, b: float) -> str:
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

def generate(seed: str):
    def scale(x, min_val, max_val):
        return min_val + x*(max_val - min_val)

    hash_value = hash(seed)
    u = ((hash_value & 0xFFF0000) >> 16)/16**3
    v = ((hash_value & 0x000FF00) >> 8)/256
    w = (hash_value &  0x00000FF)/256

    h = u * 360
    s = scale(v, 50, 100)  # Saturation between 50% and 100%
    l = scale(w, 40, 60)  # Lightness between 40% and 60%

    r, g, b = hsl2rgb(h, s, l)
    return to_hex(r, g, b)