public void calcProcessCalcPutAnnotations(int obj_40, item_93 item_20) {
  int var_94 = 2;
  int var_49 = 0;
  item_70 obj_73 = (item_70) this;
  item_70 oldest = null;
  for (; obj_73 != null; obj_73 = obj_73.previousAnnotation) {
    obj_73.executeProcessHandleVisitEnd();
    var_94 += obj_73.annotation.length;
    var_49 += 1;
    oldest = obj_73;
  }
  item_20.putShort(obj_40);
  item_20.putInt(var_94);
  item_20.putShort(var_49);
  obj_73 = oldest;
  for (; obj_73 != null; obj_73 = obj_73.nextAnnotation) {
    item_20.putByteArray(obj_73.annotation.data, 0, obj_73.annotation.length);
  }
}