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

            for (int e = exp; e > 0; e >>= 1)
            {
                if ((e & 1) != 0)
                {
                    result *= a;
                }
                a *= a;
            }

            int intResult = (int) result;
            return new IntLiteral(intResult);
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