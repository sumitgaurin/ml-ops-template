import os

def set_output(output_variable:str, output_value:any):
    """
    Sets the primitive output value  to the output variable of a component.


    :param output_variable: The output variable which needs to be set
    :param output_value: The output value which should be set to the output variable
    :return: None
    """

    # In Azure Machine Learning components, outputs of primitive types like string need to be written 
    # to a designated file path provided by the output parameter (e.g., --model_version). 
    # Azure ML reads the content of this file to set the component's output value, 
    # which can then be used in subsequent pipeline steps.
    output_file = output_variable
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_value)

def print_args(args):
    """
    Prints the arguments once they are received by the components.

    :param args: The arguments list as parsed by the argparse.ArgumentParser module 
    :return: None
    """
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")