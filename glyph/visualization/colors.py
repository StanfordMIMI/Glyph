"""Stable distinct default colors for every displayed outcome."""
import colorsys
import re

BASE_COLORS=['#0d9488','#3b82f6','#8b5cf6','#f97316','#db2777',
             '#65a30d','#dc2626','#ca8a04','#06b6d4','#92400e']


# HSLuv hues 15..303 degrees, saturation 95, lightness 60.
# Fixed perceptual lightness/saturation avoid a pale or dark end of the scale.
DIGIT_COLORS=['#fb5948','#c4831d','#9e931d','#6d9f1c','#1ea566',
              '#1fa291','#219fad','#2399d7','#957ef9','#ec3ef9']


def digit_index(name):
    value=str(name)
    match=re.fullmatch(r'(?:is[_ -]*)?digit[_ -]*([0-9])',value,re.IGNORECASE)
    return int(match.group(1)) if match else (int(value) if re.fullmatch('[0-9]',value) else None)


def outcome_palette(outcomes):
    names=sorted({str(name) for name in outcomes});result={};used=set()
    if names and all(digit_index(name) is not None for name in names):
        return {name:DIGIT_COLORS[digit_index(name)] for name in names}
    for i,name in enumerate(names):
        if i<len(BASE_COLORS):color=BASE_COLORS[i]
        else:
            phase=i*.61803398875
            while True:
                rgb=colorsys.hls_to_rgb(phase%1,.42+.12*(i%2),.70)
                color='#'+''.join(f'{round(channel*255):02x}' for channel in rgb)
                if color not in used:break
                phase+=.037
        result[name]=color;used.add(color)
    return result
