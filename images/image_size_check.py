from PIL import Image
import os
import sys


def find_smallest_resolution(directories=[]):
    if not directories:
        return None

    sorted_widths = []
    sorted_heights = []
    min_height = None
    min_width = None
    for directory in directories:
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # Check if it is a file
            if os.path.isfile(f):
                image = Image.open(f)
                if min_height is None:
                    min_height = (image.height, f)
                if min_width is None:
                    min_width = (image.width, f)

                if image.height < min_height[0]:
                    min_height = (image.height, f)
                if image.width < min_width[0]:
                    min_width = (image.width, f)
                sorted_heights.append((image.height, f))
                sorted_widths.append((image.width, f))

    sorted_heights.sort()
    sorted_widths.sort()
    # print(max(sorted_heights))
    # print(min(sorted_heights))
    # print(max(sorted_widths))
    # print(min(sorted_widths))
    print('Number of images = {len}'.format(len=len(sorted_heights)))
    print('Min width = {min_width}\nMin height = {min_height}'.format(min_width=min(sorted_widths),
                                                                      min_height=min(sorted_heights)))
    print('Max width = {max_width}\nMax height = {max_height}'.format(max_width=max(sorted_widths),
                                                                      max_height=max(sorted_heights)))
    return (min_width, min_height)


if __name__ == '__main__':
    v = sys.version_info
    print('Python version: {v0}.{v1}.{v2}'.format(v0=v[0], v1=v[1], v2=v[2]))

    directories = ['ClothMask', 'N95Mask', 'SurgicalMask', 'NoMask']
    find_smallest_resolution(directories)
