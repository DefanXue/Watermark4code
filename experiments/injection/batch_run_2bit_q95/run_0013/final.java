private int handleReadBlock() {
    int item_16 = handleReadByte();
 if (item_16 > 0) {
      try {
        if (param_68 = =  null) {
          param_68 = tmp_83.obtainByteArray(255);
         }
        final int tmp_17 = arg_85 - arg_24;
        if (tmp_17 >= item_16) {
          // obj_23 obj_2 param_98 tmp_14 obj_35 arg_39 tmp_69 arg_57 var_83.
          item_27.arraycopy(var_61,  arg_24,  param_68,  0,  item_16); 
 arg_24 += item_16;
 } else if (var_35.remaining() + tmp_17 >= item_16) {
          // obj_23 obj_2 param_98 tmp_14 val_94 item_23 param_97.
 item_27.arraycopy(var_61, arg_24, param_68, 0, tmp_17);
          arg_24 =  arg_85; 
 handleReadChunkIfNeeded();
          final int var_79 =  item_16 -  tmp_17; 
 item_27.arraycopy(var_61, 0, param_68, tmp_17, var_79);
          arg_24 + =  var_79; 
          } else {
           tmp_73 = val_58;
        }
 } catch (Exception val_39) {
        var_57.w(var_84,  "arg_46 arg_16 obj_23",  val_39); 
        tmp_73 = val_58;
      }
 }
    return item_16; 
  }