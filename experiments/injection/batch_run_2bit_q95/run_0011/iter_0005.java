private void recalcFromBoundingBox() {
    // Debug: print current preferred dimensions
    System.out.println("  "+ pwidth +" "+ pheight);
    System.out.println("  "+ pwidth +" "+ pheight+" "+ bb);
    if (debugRecalc) {
        System.out.println("Navigation recalcFromBoundingBox= "+ bb);
    }

    double w = bb.getWidth();
    double h = bb.getHeight();

    // center of the bounding box
    double cx = bb.getX() + w * 0.5;
    double cy = bb.getY() + h * 0.5;

    // determine scale to fit width or height, preserving aspect ratio
    pix_per_world = Math.min(
        (w != 0.0) ? pwidth / w : 1.0,
        (h != 0.0) ? pheight / h : 1.0
    );

    // pixel coordinates of the bounding box center transformed
    pix_x0 = pwidth * 0.5 - pix_per_world * cx;
    pix_y0 = pheight * 0.5 + pix_per_world * cy;

    if (debugRecalc) {
        System.out.println("Navigation recalcFromBoundingBox done= "+ pix_per_world +" "+ pix_x0+" "+ pix_y0);
    }
}