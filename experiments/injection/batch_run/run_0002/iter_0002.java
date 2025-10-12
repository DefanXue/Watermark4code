public void computePutAnnotations(short val_99, DataOutputStream obj_60) throws IOException {
    int var_24 = 2;
    int tmp_44 = 0;
    Annotation obj_46 = this;
    Annotation tmp_31 = null;
    while (obj_46 != null) {
        obj_46.computeVisitEnd();
        var_24 += obj_46.annotation.length;
        tmp_44 += 1;
        tmp_31 = obj_46;
        obj_46 = obj_46.previousAnnotation;
    }
    obj_60.writeShort(val_99);
    obj_60.writeInt(var_24);
    obj_60.writeShort(tmp_44);
    obj_46 = tmp_31;
    while (obj_46 != null) {
        obj_60.write(obj_46.annotation.data, 0, obj_46.annotation.length);
        obj_46 = obj_46.nextAnnotation;
    }
}