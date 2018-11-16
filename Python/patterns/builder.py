class LaptopBuilder():

    def __init__(self):
        self._laptop = Laptop()  # protected member.

    def create_laptop(self):
        self._laptop = Laptop()

    def get_laptop(self):
        return self._laptop

    # Creating a laptop: 
    def set_monitor_res(self):
        pass
    def set_proc(self):
        pass
    def set_mem(self):
        pass
    def set_hdd(self):
        pass
    def set_battery(self):
        pass

class Laptop():
    def __init__(self):
        self.monres = 0
        self.proc = 0
        self.memory = 0
        self.hdd = 0
        self.battery = 0

class GamingLaptopBuilder(LaptopBuilder):
    def set_monitor_res(self):
        self._laptop.monres = 1
    def set_proc(self):
        self._laptop.proc = 1
    def set_mem(self):
        self._laptop.memory = 1
    def set_hdd(self):
        self._laptop.hdd = 1
    def set_battery(self):
        self._laptop.battery = 1

class BusinessLaptopBuilder(LaptopBuilder):
    def set_monitor_res(self):
        self._laptop.monres = 2
    def set_proc(self):
        self._laptop.proc = 2
    def set_mem(self):
        self._laptop.memory = 2
    def set_hdd(self):
        self._laptop.hdd = 2
    def set_battery(self):
        self._laptop.battery = 2

class BuyLaptop():
    
    def __init__(self):
        self._laptopBuilder = LaptopBuilder()

    def set_laptop_builder(self, IBuilder):
        self._laptopBuilder = IBuilder

    def get_paptop(self):
        return self._laptopBuilder.get_laptop()

    def construct_laptop(self):
        self._laptopBuilder.create_laptop()
        self._laptopBuilder.set_monitor_res()
        self._laptopBuilder.set_proc()
        self._laptopBuilder.set_mem()
        self._laptopBuilder.set_hdd()
        self._laptopBuilder.set_battery()

def usage():
    businessbuilder = BusinessLaptopBuilder()
    gamebuilder = GamingLaptopBuilder()
    shop = BuyLaptop()

    shop.set_laptop_builder(gamebuilder)
    shop.construct_laptop()
    laptop = shop.get_paptop()
    print(vars(laptop))

    shop.set_laptop_builder(businessbuilder)
    shop.construct_laptop()
    laptop = shop.get_paptop()
    print(vars(laptop))

if __name__ == '__main__':
    usage()