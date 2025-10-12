private void recalcFromBoundingBox() {
    System.out.println("  "+ pwidth +" "+ pheight);
    System.out.println("  "+ pwidth +" "+ pheight+" "+ bb);
    if (debugRecalc) {
        System.out.println("Navigation recalcFromBoundingBox= "+ bb);
    }

    double w = bb.getWidth();
    double h = bb.getHeight();

    double cx = bb.getX() + w * 0.5;
    double cy = bb.getY() + h * 0.5;

    double scaleX = (w == 0.0) ? 1.0 : pwidth / w;
    double scaleY = (h == 0.0) ? 1.0 : pheight / h;
    pix_per_world = Math.min(scaleX, scaleY);

    pix_x0 = pwidth * 0.5 - pix_per_world * cx;
    pix_y0 = pheight * 0.5 + pix_per_world * cy;

    if (debugRecalc) {
        System.out.println("Navigation recalcFromBoundingBox done= "+ pix_per_world +" "+ pix_x0+" "+ pix_y0);
    }
}