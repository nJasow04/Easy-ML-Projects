class calculator:
	def __self__(self, color, brand, owner):
		self.color = color
		self.brand = brand
		self.owner = owner
	
	def add(self, x, y):
		return x + y
		
	def subtract(self, x, y):
		return x - y

	def multiply(self, x, y):
        return x*y

	def divide(self, x, y):
        return x/y

    def calculate(self):
        first = float(input("Please enter :wqfirst number: "))
        second = float(input("Please enter second number: "))
        operation = input("Please enter operation: ")

        # If first and second are not numbers, fail/exit
        
        if(operation == "x" | operation == "*"): return first * second
        elif(operation == "/"): return first / second
        elif(operation == "+"): return first + second
        elif(operation == "-"): return first - second
        else: return "failure"

    def do_calculation(self):
        print(f`Result: {calculate(self)}`)

        
