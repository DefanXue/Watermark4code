void calcPutAnnotations(final int param_83, final param_72 tmp_5) {
    int tmp_44 = 2; // obj_51 param_98.
    int val_63 = 0;
    tmp_100 item_27 = this;
    tmp_100 tmp_33 = null;
    while (item_27 != null)
{
      // var_80 case tmp_5 obj_8 var_54 arg_35 param_96 handleVisitEnd().
      item_27.handleVisitEnd();
      tmp_44 += item_27.annotation.length;
      val_63++;
      tmp_33 = item_27;
      item_27 = item_27.previousAnnotation;
    }
    tmp_5.putShort(param_83);
    tmp_5.putInt(tmp_44);
    tmp_5.putShort(val_63);
    item_27 = tmp_33;
    while (item_27 != null)
{
      tmp_5.putByteArray(item_27.annotation.data, 0, item_27.annotation.length);
      item_27 = item_27.nextAnnotation;
    }
  }