import sympy as sp

class DualNumber:
    """
    A class representing dual numbers for automatic differentiation.
    
    Attributes:
        real (float or SymPy expression): The real part of the dual number.
        dual (float or SymPy expression): The dual part representing the derivative.
    """
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return f"{self.real} + {self.dual}ε"

    # Addition
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int, float, sp.Expr)):
            return DualNumber(self.real + other, self.dual)
        else:
            return NotImplemented

    def __radd__(self, other):
        # Addition is commutative
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int, float, sp.Expr)):
            return DualNumber(self.real - other, self.dual)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, sp.Expr)):
            return DualNumber(other - self.real, -self.dual)
        else:
            return NotImplemented

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # (a + bε)(c + dε) = ac + (ad + bc)ε
            real = self.real * other.real
            dual = self.real * other.dual + self.dual * other.real
            return DualNumber(real, dual)
        elif isinstance(other, (int, float, sp.Expr)):
            return DualNumber(self.real * other, self.dual * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        # Multiplication is commutative
        return self.__mul__(other)

    # Division
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            if other.real == 0:
                raise ZeroDivisionError("Division by zero in dual numbers.")
            # (a + bε) / (c + dε) = (a/c) + ((b*c - a*d)/c^2)ε
            real = self.real / other.real
            dual = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            return DualNumber(real, dual)
        elif isinstance(other, (int, float, sp.Expr)):
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return DualNumber(self.real / other, self.dual / other)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, sp.Expr)):
            if self.real == 0:
                raise ZeroDivisionError("Division by zero in dual numbers.")
            # (c) / (a + bε) = (c/a) + (-c*b)/(a^2)ε
            real = other / self.real
            dual = (-other * self.dual) / (self.real ** 2)
            return DualNumber(real, dual)
        else:
            return NotImplemented

    # Power
    def __pow__(self, power):
        if isinstance(power, (int, float, sp.Expr)):
            if self.real == 0 and power < 0:
                raise ZeroDivisionError("0 cannot be raised to a negative power.")
            # (a + bε)^n = a^n + n*a^(n-1)*bε
            real = self.real ** power
            dual = power * (self.real ** (power - 1)) * self.dual
            return DualNumber(real, dual)
        else:
            return NotImplemented

    # Negation
    def __neg__(self):
        return DualNumber(-self.real, -self.dual)

    # Equality
    def __eq__(self, other):
        if isinstance(other, DualNumber):
            return self.real == other.real and self.dual == other.dual
        elif isinstance(other, (int, float, sp.Expr)):
            return self.real == other and self.dual == 0
        else:
            return False

    # Implement additional mathematical functions as methods
    def sin(self):
        # sin(a + bε) = sin(a) + b*cos(a)ε
        return DualNumber(sp.sin(self.real), self.dual * sp.cos(self.real))

    def cos(self):
        # cos(a + bε) = cos(a) - b*sin(a)ε
        return DualNumber(sp.cos(self.real), -self.dual * sp.sin(self.real))

    def tan(self):
        # tan(a + bε) = tan(a) + b*(sec(a)^2)ε
        return DualNumber(sp.tan(self.real), self.dual * (1 / sp.cos(self.real))**2)

    def exp(self):
        # exp(a + bε) = exp(a) + b*exp(a)ε
        exp_real = sp.exp(self.real)
        return DualNumber(exp_real, self.dual * exp_real)

    def log(self):
        # log(a + bε) = log(a) + (b/a)ε
        if self.real <= 0:
            raise ValueError("Log undefined for non-positive real part.")
        return DualNumber(sp.log(self.real), self.dual / self.real)

    def sqrt(self):
        # sqrt(a + bε) = sqrt(a) + (b)/(2*sqrt(a))ε
        if self.real < 0:
            raise ValueError("Square root undefined for negative real part.")
        sqrt_real = sp.sqrt(self.real)
        return DualNumber(sqrt_real, self.dual / (2 * sqrt_real))

    # You can add more functions as needed (e.g., sinh, cosh, etc.)

# Optional: Define external functions to handle DualNumbers
def dual_sin(dn):
    if isinstance(dn, DualNumber):
        return dn.sin()
    else:
        return sp.sin(dn)

def dual_cos(dn):
    if isinstance(dn, DualNumber):
        return dn.cos()
    else:
        return sp.cos(dn)

def dual_tan(dn):
    if isinstance(dn, DualNumber):
        return dn.tan()
    else:
        return sp.tan(dn)

def dual_exp(dn):
    if isinstance(dn, DualNumber):
        return dn.exp()
    else:
        return sp.exp(dn)

def dual_log(dn):
    if isinstance(dn, DualNumber):
        return dn.log()
    else:
        return sp.log(dn)

def dual_sqrt(dn):
    if isinstance(dn, DualNumber):
        return dn.sqrt()
    else:
        return sp.sqrt(dn)
