import math

class MinerAgent:
    ACTIVE="ACTIVE"
    OFFLINE="OFFLINE"

    def __init__(
        self,
        miner_id:int,
        capacity:float,
        unit_cost:float,
    ):
        if capacity <= 0:
            raise ValueError("capacity phải lớn hơn 0")
        if unit_cost < 0:
            raise ValueError("unit_cost không được âm")

        self.miner_id=miner_id
        self.capacity=capacity
        self.unit_cost=unit_cost

        self.reward = 0.0
        self.profit = 0.0
        self.status = self.ACTIVE
        self.datasets = set()

    #tinh capacity 
    @property
    def used_capacity(self):
        # Stable summation order helps identical seeds reproduce identical results.
        return sum(
            dataset.size
            for dataset in sorted(self.datasets, key=lambda item: item.dataset_id)
        )
    @property
    def free_capacity(self):
        return self.capacity-self.used_capacity

    #them/xoa dataset
    def add_dataset(self,dataset):
        if dataset in self.datasets:
            return False
        if dataset.size>self.free_capacity:
            return False

        self.datasets.add(dataset)
        dataset.add_miner(self)
        return True

    def remove_dataset(self,dataset):
        if dataset not in self.datasets:
            return False

        self.datasets.remove(dataset)
        dataset.remove_miner(self)
        return True

    #tinh reward,cost,profit

    def calculate_operating_cost(self):
        return self.unit_cost*self.used_capacity

    def calculate_reward(self,storage_price:float):
        if storage_price<0:
            raise ValueError("storage price khong am")
        return storage_price*self.used_capacity

    def calculate_profit(self,storage_price:float):
        self.reward=self.calculate_reward(storage_price)
        self.profit=self.reward-self.calculate_operating_cost()
        return self.profit

    #xac xuat roi mang
    def leave_probability(self,beta:float):
        if beta<0:
            raise ValueError("bata khong am")

        x=beta*self.profit
        if x>=0:
            exp_neg_x = math.exp(-x)
            return exp_neg_x/(1.0+exp_neg_x)
        return 1.0/(1.0+math.exp(x))

    def go_offline(self) -> None:
        self.status = self.OFFLINE
        

        



        
        
