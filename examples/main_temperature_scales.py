from coordinatus import Space, Space1D, Point
from coordinatus.transforms import translate1D, scale1D

# Kelvin is the 1D world space (absolute)
kelvin = Space1D()

# ts1D(tx, sx) maps x → sx·x + tx
celsius    = Space(transform=translate1D(273.15), parent=kelvin)   # K = C + 273.15
fahrenheit = Space(transform=scale1D(5/9) @ translate1D(-32), parent=celsius)   # C = (F - 32) * 5/9

absolute_zero = Point([0.0], space=kelvin)  # 0 K
absolute_zero_k = absolute_zero.relative_to(kelvin).coords[0]
absolute_zero_c = absolute_zero.relative_to(celsius).coords[0]
absolute_zero_f = absolute_zero.relative_to(fahrenheit).coords[0]
print(f"Absolute zero:        \t{absolute_zero_k:.1f} K\t{absolute_zero_c:.1f} °C\t{absolute_zero_f:.1f} °F")

freezing_temp = Point([0.0], space=celsius)  # 0.0 °C
freezing_temp_k = freezing_temp.relative_to(kelvin).coords[0]
freezing_temp_c = freezing_temp.relative_to(celsius).coords[0]
freezing_temp_f = freezing_temp.relative_to(fahrenheit).coords[0]
print(f"Freezing temperature: \t{freezing_temp_k:.1f} K\t{freezing_temp_c:.1f} °C\t{freezing_temp_f:.1f} °F")

human_body_temp = Point([100.0], space=fahrenheit)  # 100 °F
human_body_temp_k = human_body_temp.relative_to(kelvin).coords[0]
human_body_temp_c = human_body_temp.relative_to(celsius).coords[0]
human_body_temp_f = human_body_temp.relative_to(fahrenheit).coords[0]
print(f"Human body temperature:\t{human_body_temp_k:.1f} K\t{human_body_temp_c:.1f} °C\t{human_body_temp_f:.1f} °F")

boiling_temp = Point([100.0], space=celsius)  # 100 °C
boiling_temp_k = boiling_temp.relative_to(kelvin).coords[0]
boiling_temp_c = boiling_temp.relative_to(celsius).coords[0]
boiling_temp_f = boiling_temp.relative_to(fahrenheit).coords[0]
print(f"Boiling temperature:   \t{boiling_temp_k:.1f} K\t{boiling_temp_c:.1f} °C\t{boiling_temp_f:.1f} °F")

