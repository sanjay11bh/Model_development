class calculator:

    def add(self , num1 , num2 ):
        print(f"addition of two number is :")
        return num1+num2
    def subtract(self , num1 , num2 ):
            print(f"subtraction of two number is :")
            return num1-num2

cal = calculator()
print("Sum:", cal.add(5, 3))
print("Difference:", cal.subtract(5, 3))    