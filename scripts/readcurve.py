import numpy as np

def read_visit_curve_file(filename):
    data = {
        "time": None,
        "functions": {}
    }

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse time
        if line.startswith("# TIME"):
            try:
                data["time"] = float(line.split()[2])
            except (IndexError, ValueError):
                raise ValueError(f"Could not parse time from line: {line}")
            i += 1
            continue

        # Parse function data
        if line.startswith("#"):
            func_name = line[1:].strip()
            x_vals = []
            y_vals = []
            i += 1

            # Read until next comment or end of file
            while i < len(lines) and not lines[i].strip().startswith("#"):
                tokens = lines[i].strip().split()
                if len(tokens) == 2:
                    try:
                        x, y = float(tokens[0]), float(tokens[1])
                        x_vals.append(x)
                        y_vals.append(y)
                    except ValueError:
                        raise ValueError(f"Could not parse x,y from line: {lines[i]}")
                i += 1

            data["functions"][func_name] = (np.array(x_vals), np.array(y_vals))
        else:
            i += 1

    return data

