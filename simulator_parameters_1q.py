def get_parameter(parameter):
    parameter_array = ['freq', 'amp', 'beta', 'ef_freq', 'ef_amp', 'ef_beta', 'dt']
    if parameter not in parameter_array:
        raise Exception(f"""The given parameter {parameter} is not a part of simulator parameter.
                        The allowed parameters are {parameter_array}.""")
    parameter_dict = {
        'freq': 5e9,
        'amp': 0.059559844136557474,
        'beta': -1.02020202020202,
        'ef_freq': 4.7e9,
        'ef_amp': 0.04244432269595312,
        'ef_beta': -3.9494949494949494
    }
    
    dt = 1/4.5e9
    
    if parameter == 'dt':
        return dt
    else:
        return parameter_dict[parameter]