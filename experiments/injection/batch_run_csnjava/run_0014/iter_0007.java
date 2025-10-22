protected NumericType evaluate(NumericType first, NumericType second)
{
    boolean bothIntegers = first.isInteger() && second.isInteger();

    if (bothIntegers)
    {
        int base = first.intValue();
        int exp = second.intValue();

        if (exp > 0)
        {
            long result = 1L;
            long a = base;

            int e = exp;
            while (e > 0)
            {
                if ((e & 1) != 0)
                {
                    result *= a;
                }
                a *= a;
                e >>= 1;
            }

            return new IntLiteral((int) result);
        }
        else
        {
            return new IntLiteral(1);
        }
    }
    else
    {
        double a = first.doubleValue();
        double b = second.doubleValue();
        return new DoubleLiteral(Math.pow(a, b));
    }
}