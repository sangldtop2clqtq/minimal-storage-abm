import math

def bonding_curve(
        rho:float,
        rho0:float,
        p_min:float,
        p_max:float,
        k:float,
):
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho phải nằm trong [0, 1]")
    if p_min < 0 or p_max < p_min:
        raise ValueError("Cần 0 <= p_min <= p_max")
    if k <= 0:
        raise ValueError("k phải lớn hơn 0")
    if not 0.0 <= rho0 <= 1.0:
        raise ValueError("rho0 phải nằm trong [0, 1]")
    exponent=-k*(rho-rho0)
    return p_min+(p_max-p_min)/(1+math.exp(exponent))

def repair_reward(
        deficit:int,
        target_replicas: int,
        base_reward: float,
        alpha:float,
):
    if target_replicas <= 0:
        raise ValueError("target_replicas phải lớn hơn 0")
    if deficit < 0 or deficit > target_replicas:
        raise ValueError("deficit phải nằm trong [0, target_replicas]")
    if base_reward < 0:
        raise ValueError("base_reward không được âm")
    if alpha < 0:
        raise ValueError("alpha không được âm")
    
    return base_reward*(1+alpha*deficit/target_replicas)

def repair_probability(utility:float,beta:float):
    if beta < 0:
        raise ValueError("beta không âm")
    x=beta*utility

    if x>=0:
        return 1.0/(1.0+math.exp(-x))

    exp_x=math.exp(x)
    return exp_x/(1.0+exp_x)

    
