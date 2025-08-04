use ndarray::{Array1, Array2, Array3};

use metastructures::topology::force_geodesic::ForceGeodesic;
use plotly::{Plot, Scatter, common::Mode};

// if i set all w_k to be the same value
// and set all w_a to be the same value
// and feed the reservoir constant data
// i can test the mechanics by watching all surface nodes becoming equidistant,
// the centroid should gravitate towards zero on all dimensions.

fn main() {
    let mut h = ForceGeodesic::new(3, 3, 24, 3);
    h.w_k = Array3::ones((24, 24, 3));
    h.w_a = Array2::ones((24, 3));

    let mut cents = vec![];

    for i in 0..100000 {
        let c = h.forward(Array1::ones(3), 0.00001);
        let ma = c.mean().unwrap();

        if i % 100 == 0 && i > 0 {
            cents.push(ma);
        }
    }

    let mut plot = Plot::new();
    let y = (0..cents.len()).collect::<Vec<usize>>();
    let pred_trace = Scatter::new(y, cents).mode(Mode::Lines);
    plot.add_trace(pred_trace);
    plot.show();
}
