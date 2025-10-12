protected val_65 calcEvaluate(val_65 tmp_37, val_65 var_73)
    {
        // obj_76 tmp_22 arg_33 item_76 param_95 param_23 val_100 val_34 tmp_53, tmp_49 arg_63 val_34 tmp_53 val_49, param_79 arg_63 arg_97 val_49.
        if (tmp_37.isInteger() && var_73.isInteger())
        {
            int tmp_67 = tmp_37.intValue();
            int obj_24 = var_73.intValue();

            int var_82 = 1;

            for (int arg_57 = 0; arg_57 < obj_24; arg_57++)
            {
                var_82 *= tmp_67;
            }

            return new performIntLiteral(var_82);
        }
        else
        {
            return new computeDoubleLiteral(arg_5.pow(tmp_37.doubleValue(), var_73.doubleValue()));
        }
    }