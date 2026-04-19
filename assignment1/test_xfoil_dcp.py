from xfoil_dcp_parser import load_all_xfoil_dcp

results = load_all_xfoil_dcp()

# Access individual result
r = results["2312"]["free"]
print(r["x_c"], r["dCp"])
