
def print_latex_table(airfoils, method_labels, data_attr, caption, label, fmt=".2f"):
    """
    Print a LaTeX table for airfoil data.

    Args:
        airfoils: dict of airfoil objects keyed by code
        method_labels: list of method label strings
        data_attr: attribute name on airfoil (e.g. 'cl_slopes_dict' or 'cl_offsets_dict')
        caption: LaTeX table caption string
        label: LaTeX table label string
        fmt: format string for values (default ".2f")
    """
    available_methods = [
        m for m in method_labels
        if any(getattr(af, data_attr).get(m) is not None for af in airfoils.values())
    ]

    header = " & ".join(["Airfoil"] + available_methods) + r" \\"
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l" + "c" * len(available_methods) + "}")
    print(r"\hline")
    print(header)
    print(r"\hline")
    for code, af in airfoils.items():
        row = [f"{code}"]
        for method in available_methods:
            value = getattr(af, data_attr).get(method)
            row.append(f"{value:{fmt}}" if value is not None else "-")
        print(" & ".join(row) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\end{table}")

