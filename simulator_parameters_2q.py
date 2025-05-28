def get_parameter(parameter, qutrit):
    parameter_array = ['freq', 'amp', 'beta', 'ef_freq', 'ef_amp', 'ef_beta', 'dt']
    if parameter not in parameter_array:
        raise Exception(f"""The given parameter {parameter} is not a part of simulator parameter.
                        The allowed parameters are {parameter_array}.""")
    parameter_dict_0 = {
        'freq': 4.9e9,
        'amp': 0.02114696903033394,
        'beta': -1.150753768844221,
        'ef_freq': 4.5e9,
        'ef_amp': 0.014512435949889333,
        'ef_beta': 0.3838383838383841
    }
    parameter_dict_1 = {
        'freq': 5.5e9,
        'amp': 0.02048768647421635,
        'beta': -1.8775510204081636,
        'ef_freq': 5.2e9,
        'ef_amp': 0.014541354298407676,
        'ef_beta': -0.9696969696969697
    }
    
    dt = 1/4.5e9
    
    if parameter == 'dt':
        return dt
    elif qutrit == 0:
        return parameter_dict_0[parameter]
    elif qutrit == 1:
        return parameter_dict_1[parameter]