from qiskit import pulse

dchan0 = pulse.DriveChannel(0)
dchan1 = pulse.DriveChannel(1)

uchan0 = pulse.ControlChannel(0)
uchan1 = pulse.ControlChannel(1)

achan0 = pulse.AcquireChannel(0)
achan1 = pulse.AcquireChannel(1)

memslot0 = pulse.MemorySlot(0)
memslot1 = pulse.MemorySlot(1)

def get_i_pulse_0():
    measure_delay = 320
    with pulse.build(name='i') as ipulse:
        pulse.delay(measure_delay, achan0)
        pulse.acquire(1, achan0,memslot0)
    
    return ipulse

def get_meas_basis_pulse_0(sub, sdg, measure):
    if measure:
        measure_delay = 3*320
    if sub not in ['01', '12', '02']:
        raise Exception(f"Subspace value error. {sub} value provided not '01' or '12'.")
    with pulse.build(name='h_'+sub) as hpulse:
        if sub=='01':
            if not sdg:
                pulse.set_frequency(freq_0, dchan0)
                pulse.shift_phase(np.pi/2, dchan0)
                pulse.play(pulse.Gaussian(320, amp_0/2, 80), dchan0)
                pulse.shift_phase(np.pi/2, dchan0)
                
                pulse.delay(measure_delay, achan0)
                pulse.acquire(1, achan0, memslot0)
            else:
                pulse.shift_phase(np.pi/2, dchan0) 
                pulse.set_frequency(freq_0, dchan0)
                pulse.shift_phase(np.pi/2, dchan0)
                pulse.play(pulse.Gaussian(320, amp_0/2, 80), dchan0)
                pulse.shift_phase(np.pi/2, dchan0)
                
                pulse.delay(measure_delay, achan0)
                pulse.acquire(1, achan0, memslot0)
        elif sub=='02':
            if not sdg:
                ren_ph = 1
                ren_amp =1.07
                pulse.set_frequency(ef_freq_0, dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*ef_amp_0, 80, ef_beta_0), dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)

                ren_ph = 1.41
                ren_amp =1.07
                pulse.set_frequency(freq_0, dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*amp_0/2, 80, beta_0), dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)

                ren_ph = 1
                ren_amp =1.
                pulse.set_frequency(ef_freq_0, dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*ef_amp_0, 80, ef_beta_0), dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                
                pulse.delay(measure_delay, achan0)
                pulse.acquire(1, achan0, memslot0)
            else:
                ren_ph = 1
                ren_amp =1.07
                pulse.set_frequency(ef_freq_0, dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*ef_amp_0, 80, ef_beta_0), dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)

                cor_sdg = 1
                pulse.shift_phase(-cor_sdg*np.pi/2, dchan0)

                ren_ph = 1.41
                ren_amp =1.07
                pulse.set_frequency(freq_0, dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*amp_0/2, 80, beta_0), dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)

                cor_sdg = 1
                pulse.shift_phase(cor_sdg*np.pi/2, dchan0)

                ren_ph = 1
                ren_amp =1.
                pulse.set_frequency(ef_freq_0, dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*ef_amp_0, 80, ef_beta_0), dchan0)
                pulse.shift_phase(-ren_ph*np.pi/2, dchan0)
                
                pulse.delay(measure_delay, achan0)
                pulse.acquire(1, achan0, memslot0)
        elif sub=='12':
            if not sdg:
                ren_ph = 0.82
                ren_amp =1.09
                pulse.set_frequency(ef_freq_0, dchan0)
                pulse.shift_phase(ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*ef_amp_0/2, 80, beta_0), dchan0)
                pulse.shift_phase(ren_ph*np.pi/2, dchan0)
                
                pulse.delay(measure_delay, achan0)
                pulse.acquire(1, achan0, memslot0)
            else:
                cor_sdg = 1.
                pulse.shift_phase(-cor_sdg*np.pi/2, dchan0)

                ren_ph = 0.82
                ren_amp =1.09
                pulse.set_frequency(ef_freq_0, dchan0)
                pulse.shift_phase(ren_ph*np.pi/2, dchan0)
                pulse.play(pulse.Drag(320, ren_amp*ef_amp_0/2, 80, beta_0), dchan0)
                pulse.shift_phase(ren_ph*np.pi/2, dchan0)
                
                pulse.delay(measure_delay, achan0)
                pulse.acquire(1, achan0, memslot0)
        
        return hpulse


def get_i_pulse_1():
    measure_delay = 320
    with pulse.build(name='i') as ipulse:
        pulse.delay(measure_delay, achan1)
        pulse.acquire(1, achan1, memslot1)
    
    return ipulse

def get_meas_basis_pulse_1(sub, sdg, measure):
    if measure:
        measure_delay = 3*320
    if sub not in ['01', '12', '02']:
        raise Exception(f"Subspace value error. {sub} value provided not '01' or '12'.")
    with pulse.build(name='h_'+sub) as hpulse:
        if sub=='01':
            if not sdg:
                ren_ph_2 = 1      
                ren_amp_2 = 1   
                pulse.set_frequency(freq_1, dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_2*amp_1/2, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                
                pulse.delay(measure_delay, achan1)
                pulse.acquire(1, achan1, memslot1)
            else:
                cor_sdg = 1.083
                pulse.shift_phase(-cor_sdg*np.pi/2, dchan1)

                ren_ph_2 = 1.05      
                ren_amp_2 = 1   
                pulse.set_frequency(freq_1, dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_2*amp_1/2, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                
                pulse.delay(measure_delay, achan1)
                pulse.acquire(1, achan1, memslot1)
        elif sub=='02':
            if not sdg:
                ren_ph_1 = 1      
                ren_amp_1 = 1   
                pulse.set_frequency(ef_freq_1, dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_1*ef_amp_1, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)

                ren_ph_2 = 0.3      
                ren_amp_2 = 1   
                pulse.set_frequency(freq_1, dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_2*amp_1/2, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)

                ren_ph_1 = 1      
                ren_amp_1 = 1   
                pulse.set_frequency(ef_freq_1, dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_1*ef_amp_1, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                
                pulse.delay(measure_delay, achan1)
                pulse.acquire(1, achan1, memslot1)
            else:
                cor_sdg = 1.083
                pulse.shift_phase(-cor_sdg*np.pi/2, dchan1)

                ren_ph_1 = 1      
                ren_amp_1 = 1   
                pulse.set_frequency(ef_freq_1, dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_1*ef_amp_1, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)

                ren_ph_2 = 1.4     
                ren_amp_2 = 1   
                pulse.set_frequency(freq_1, dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_2*amp_1/2, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)

                ren_ph_1 = 1      
                ren_amp_1 = 1   
                pulse.set_frequency(ef_freq_1, dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_1*ef_amp_1, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                
                pulse.delay(measure_delay, achan1)
                pulse.acquire(1, achan1, memslot1)
        elif sub=='12':
            if not sdg:
                ren_ph_1 = 0.97      
                ren_amp_1 = 0.99   
                pulse.set_frequency(ef_freq_1, dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_1*ef_amp_1/2, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)

                ren_ph_2 = 0     
                ren_amp_2 = 1   
                pulse.set_frequency(freq_1, dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                pulse.play(pulse.Drag(2, ren_amp_2*amp_1/2, 1, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                
                pulse.delay(measure_delay, achan1)
                pulse.acquire(1, achan1, memslot1)
            else:
                cor_sdg = 1.083
                pulse.shift_phase(-cor_sdg*np.pi/2, dchan1)

                ren_ph_1 = 1.03      
                ren_amp_1 = 1   
                pulse.set_frequency(ef_freq_1, dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)
                pulse.play(pulse.Drag(320, ren_amp_1*ef_amp_1/2, 80, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_1*np.pi/2, dchan1)

                ren_ph_2 = 0     
                ren_amp_2 = 1   
                pulse.set_frequency(freq_1, dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                pulse.play(pulse.Drag(2, ren_amp_2*amp_1/2, 1, ef_beta_1), dchan1)
                pulse.shift_phase(ren_ph_2*np.pi/2, dchan1)
                
                pulse.delay(measure_delay, achan1)
                pulse.acquire(1, achan1, memslot1)
        
        return hpulse