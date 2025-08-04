use ndarray::{Array1, Array2, Array3, ArrayView, Axis, Ix1};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_stats::QuantileExt;

pub struct ForceGeodesic {
    pub d_in: usize,
    pub d_model: usize,
    pub n_nodes: usize,
    pub d_out: usize,

    pub surface_nodes: Array2<f64>,

    pub w_affine: Array2<f64>,
    pub w_a: Array2<f64>,
    pub w_k: Array3<f64>,
    pub w_out: Array2<f64>,

    pub epsilon: f64,
}

impl ForceGeodesic {
    pub fn new(d_in: usize, d_model: usize, n_nodes: usize, d_out: usize) -> ForceGeodesic {
        ForceGeodesic {
            d_in,
            d_model,
            n_nodes,
            d_out,

            surface_nodes: ForceGeodesic::random_face_nodes(d_model, n_nodes),

            w_affine: ForceGeodesic::he((d_in, d_model)),
            w_a: ForceGeodesic::softmax(ForceGeodesic::he((n_nodes, d_model))),
            w_k: ForceGeodesic::random_k(d_model, n_nodes),
            w_out: ForceGeodesic::he((d_model, d_out)),

            epsilon: 1e-4,
        }
    }

    pub fn set_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    pub fn l2(x: ArrayView<f64, Ix1>) -> f64 {
        x.map(|x| x.powi(2)).sum().sqrt()
    }

    pub fn he(shape: (usize, usize)) -> Array2<f64> {
        let bound = f64::sqrt(2.) / f64::sqrt(shape.0 as f64);
        return Array2::random(shape, Uniform::new(-bound, bound));
    }

    pub fn softmax(x: Array2<f64>) -> Array2<f64> {
        let maxes = x
            .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
            .insert_axis(Axis(1));

        let mut d = x - &maxes;

        d.mapv_inplace(|x| x.exp());

        let sums = d.map_axis(Axis(1), |row| row.sum()).insert_axis(Axis(1));

        let last = &d / &sums;
        return last;
    }

    pub fn random_face_nodes(d_model: usize, n_nodes: usize) -> Array2<f64> {
        let mut v: Array2<f64> = Array2::random((n_nodes, d_model), Uniform::new(-1., 1.));
        for mut row in v.rows_mut() {
            row.assign(&(&row / ForceGeodesic::l2(row.view())));
        }

        v
    }

    pub fn random_k(d_model: usize, n_nodes: usize) -> Array3<f64> {
        let v: Array3<f64> = Array3::random((n_nodes, n_nodes, d_model), Uniform::new(-1., 1.));
        v
    }

    pub fn forward(&mut self, x: Array1<f64>, step_size: f64) -> Array2<f64> {
        // Problem
        // w_affine: linear transformation of d_in to d_model
        // w_a: describes what portion of the input energy a node will receive
        // w_k: describes per each node, per each dimension, how strongly this node repels a peer
        // along that model dimension
        //
        // I need to vectorize the following operation
        //
        // - Take an input vector of d_in and transform it into d_model
        // - Divide its energy along all surface nodes, using w_a
        // - For each subdivision of energy per node, create a matrix, f_k, with the node -> all other
        // node forces resulting from a combination of energy resulting from w_a (n_nodes, d_model) and the force
        // weights w_k. f_k = (n_nodes * n_nodes, d_model)
        // - Calculate a distance matrix, a_d, that contains each nodes angular
        // distance to each other node (n_nodes, n_nodes)
        // - Use a_d and f_k as inputs to the inverse square law to create a force matrix isl_k
        // (n_nodes * n_nodes, d_model) that describes how much force each node pushes every other
        // node with
        // - Create a matrix of directed tangents from each node to each other node, describing
        // which direction the force should be applied d_f (n_nodes, n_nodes, d_model)
        // - Given directions d_f (n_nodes, n_nodes, d_model), forces isl_k (n_nodes, n_nodes, d_model),
        // and positions surface_nodes (n_nodes, d_model), create a matrix of geodesic positional
        // updates g_u (n_nodes, d_model) that describes the systems next state
        // - Create a vector c (d_model) that describes the centroid of the updated positions
        // - That is my output

        let x = x
            .insert_axis(Axis(0))
            .dot(&self.w_affine)
            .remove_axis(Axis(0)); // (, d_model)
        let x_split = &self.w_a * &x; // (n_nodes, d_model)

        // broadcast x_split into shape (n_nodes, n_nodes, d_model) along the middle axis, such
        // that the middle axis is a repeat of x_split per that node. then i just have to
        // elementwise multiply x_split_broadcast against w_k to get the weight specific forces
        // f_k (n_nodes, n_nodes, d_model)

        let x_split_broadcast = x_split
            .insert_axis(Axis(1))
            .broadcast((self.n_nodes, self.n_nodes, self.d_model))
            .unwrap()
            .to_owned();
        let f_k = &x_split_broadcast * &self.w_k;

        // calculate inter-node distances by
        // a_d = acos(s_n • s_n.T) = (n_nodes, n_nodes)
        let a_d = self
            .surface_nodes
            .dot(&self.surface_nodes.t())
            .clamp(-1., 1.)
            .mapv(|x| x.acos());

        // apply inverse square law
        // convert a_d into the denominator
        let a_d_sq_eps = a_d.mapv(|x| x.powi(2) + 1e-4);

        // gonna have to broadcast a_d into a shape divisible by f_k
        // which would be - if f_k is (n_nodes * n_nodes, d_model)
        // then i would want a_d tp be (n_nodes * n_nodes)
        let a_d_reshape = a_d_sq_eps
            .insert_axis(Axis(2))
            .broadcast((self.n_nodes, self.n_nodes, self.d_model))
            .unwrap()
            .to_owned();

        // compute the inverse square law

        let isl_k = &f_k / &a_d_reshape; // (n_nodes, n_nodes, d_model)

        // now i need to calculate directions from each node to each other node, will be another
        // dot product
        // vD = y - (y•x) * x
        // surface_nodes = s_d = (n_nodes, d_model)
        // dot = s_d • s_d.T = (n_nodes, n_nodes)

        let dot = self
            .surface_nodes
            .dot(&self.surface_nodes.t())
            .insert_axis(Axis(2)); // (n_nodes, n_nodes)

        // broadcast surface_nodes into:
        // s_i = (n_nodes, n_nodes, d_model) with fixed row variable column
        // s_j = (n_nodes, n_nodes, d_model) with fixed column variable row
        // for full pairwise comparison

        let s_i = self
            .surface_nodes
            .view()
            .insert_axis(Axis(1))
            .broadcast((self.n_nodes, self.n_nodes, self.d_model))
            .unwrap()
            .to_owned();
        let s_j = self
            .surface_nodes
            .view()
            .insert_axis(Axis(0))
            .broadcast((self.n_nodes, self.n_nodes, self.d_model))
            .unwrap()
            .to_owned();

        // calculate v = s_i - (dot * s_j)
        let v = s_j - (&dot * &s_i); // (n_nodes, n_nodes, d_model)

        // need to normalize axis 2 with l2 norm, then divide Axis(2) of original by this norm
        // d_d = v / ||v||
        let v_norm = v.map_axis(Axis(2), |x| ForceGeodesic::l2(x)); // (n_nodes, n_nodes)

        let mut mask = Array2::zeros((self.n_nodes, self.n_nodes));
        for i in 0..self.n_nodes {
            mask[[i, i]] = f64::INFINITY;
        }

        let v_norm = v_norm + mask;

        // need to mask out the diagonal of v_norm somehow.

        let v_norm_broadcast = v_norm
            .insert_axis(Axis(2))
            .broadcast((self.n_nodes, self.n_nodes, self.d_model))
            .unwrap()
            .to_owned();
        let d_d = &v / &v_norm_broadcast; // (n_nodes, n_nodes, d_model)

        // sum all forces together
        let f = (isl_k * d_d).sum_axis(Axis(0)); // (n_nodes, d_model

        // a is the sum force to be used in geodesic update
        let a = f
            .map_axis(Axis(1), |x| ForceGeodesic::l2(x))
            .insert_axis(Axis(1)); // (d_model, 1)

        // now to find d, direction to be used in geodesic update
        // need to turn F directions into tangents along surface_nodes
        // coming from F to surface node, its 1:1
        // v = y - (y•x) * x

        let dot = self
            .surface_nodes
            .axis_iter(Axis(0))
            .zip(f.axis_iter(Axis(0)))
            .map(|(s, x)| s.dot(&x))
            .collect::<Array1<f64>>(); // (n_nodes, d_model)

        let d = &f - (&dot.insert_axis(Axis(1)) * &self.surface_nodes);
        let d_norm = d
            .map_axis(Axis(1), |x| ForceGeodesic::l2(x))
            .insert_axis(Axis(1));
        let d = &d / &d_norm;

        // apply the geodesic update
        self.surface_nodes =
            (&a * step_size).cos() * &self.surface_nodes + (&a * step_size).sin() * &d;

        // find the centroid -> project it to d_out as the output of the system

        let centroid = self
            .surface_nodes
            .mean_axis(Axis(0))
            .unwrap()
            .insert_axis(Axis(0));
        let out = centroid.dot(&self.w_out);

        out
    }
}

#[test]
fn nodes_remain_on_sphere() {
    let mut h = ForceGeodesic::new(3, 5, 12, 3);

    for _ in 0..100 {
        let x = Array1::random(3, Uniform::new(0.0, 1.0));
        h.forward(x, 0.1);
    }

    for pos in h.surface_nodes.axis_iter(Axis(0)) {
        let l2 = ForceGeodesic::l2(pos).abs();
        println!("{}", l2);
    }

    let all_on_surface = h.surface_nodes.axis_iter(Axis(0)).all(|x| {
        let l2 = ForceGeodesic::l2(x).abs();
        1. - l2 < 1e-4
    });

    assert!(all_on_surface);
}
