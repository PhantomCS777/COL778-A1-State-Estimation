import json

files = ["state_estimation_part_1.ipynb","state_estimation_part_2.ipynb"]
output_file = ["A1.py"]
py_file = open(f"{output_file[0]}", "w+")
for i,file in enumerate(files):
    code = json.load(open(file))
    for cell in code['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                line = line.replace("fig.show()","# fig.show()")
                py_file.write(line)
            py_file.write("\n")
        elif cell['cell_type'] == 'markdown':
            py_file.write("\n")
            for line in cell['source']:
                if line and line[0] == "#":
                    py_file.write(line)
            py_file.write("\n")

py_file.close()