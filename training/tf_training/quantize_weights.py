#!/usr/bin/env python3
import sys
import os
import argparse


def format_n(x):
    x = float(x)
    x = '{:.3g}'.format(x)
    x = x.replace('e-0', 'e-')
    if x.startswith('0.'):
        x = x[1:]
    if x.startswith('-0.'):
        x = '-' + x[2:]
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Quantize network file to decrease the file size.')
    parser.add_argument("input", help='Input file', type=str)
    parser.add_argument("-o", "--output", required=False, type=str, default=None,
                        help='Output file. Defaults to input + "_quantized"')
    args = parser.parse_args()

    if args.output is None:
        output_name = os.path.splitext(sys.argv[1])
        output_name = output_name[0] + '_quantized' + output_name[1]
    else:
        output_name = args.output
    print('Writing to output file: {}'.format(output_name))
    output = open(output_name, 'w')

    calculate_error = True
    error = 0

    with open(args.input, 'r') as f:
        for line in f:
            line = line.split(' ')
            lineq = list(map(format_n, line))

            if calculate_error:
                e = sum((float(line[i]) - float(lineq[i]))**2 for i in range(len(line)))
                error += e / len(line)
            output.write(' '.join(lineq) + '\n')

    if calculate_error:
        print('Weight file difference L2-norm: {}'.format(error**0.5))

    output.close()
