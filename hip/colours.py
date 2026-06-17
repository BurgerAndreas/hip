import seaborn as sns

HESSIAN_METHOD_TO_COLOUR = {
    "autograd": "#5e859e", '#a1c9f4'
    "autograd_conservative": "#b482c8",
    "forward_pass": "#8ed3c3",
    "finite_difference_bz1": "#837d80",
    "finite_difference_bz32": "#ffa8af",
    "prediction": "#d96001", # "#ffb482" #ae5a41 #db95a6
    "ef": "#837d80", # "#5a5255",  #debb9b
}
HESSIAN_METHOD_TO_COLOUR["predict"] = HESSIAN_METHOD_TO_COLOUR["prediction"]
HESSIAN_METHOD_TO_COLOUR["learned"] = HESSIAN_METHOD_TO_COLOUR["prediction"]
HESSIAN_METHOD_TO_COLOUR["hip"] = HESSIAN_METHOD_TO_COLOUR["prediction"]

# Relaxations
OPTIM_TO_COLOUR = {
    "firstorder": "#295c7e",
    "bfgs": "#636EFA",
    "secondorder": "#db95a6",
}
OPTIM_TO_COLOUR["First-Order"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["Second-Order"] = OPTIM_TO_COLOUR["secondorder"]
OPTIM_TO_COLOUR["Quasi-Second-Order"] = OPTIM_TO_COLOUR["bfgs"]
OPTIM_TO_COLOUR["No Hessian"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["No Hessians"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["Hessian Free"] = OPTIM_TO_COLOUR["firstorder"]
OPTIM_TO_COLOUR["Quasi-Hessian"] = OPTIM_TO_COLOUR["bfgs"]
OPTIM_TO_COLOUR["Hessian"] = OPTIM_TO_COLOUR["secondorder"]


# ANNOTATION_FONT_SIZE = 16
# ANNOTATION_BOLD_FONT_SIZE = 18
# AXES_FONT_SIZE = 14
# AXES_TITLE_FONT_SIZE = 16
# LEGEND_FONT_SIZE = 16
# TITLE_FONT_SIZE = 20

ANNOTATION_BOLD_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 14
AXES_FONT_SIZE = 12
AXES_TITLE_FONT_SIZE = 13
LEGEND_FONT_SIZE = 12
TITLE_FONT_SIZE = 16