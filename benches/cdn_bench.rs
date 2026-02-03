use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use alice_cdn::prelude::*;
use alice_cdn::{ContentId, NodeId, SMALL_TABLE_SIZE, DEFAULT_TABLE_SIZE};

fn bench_simd_distance(c: &mut Criterion) {
    let a = SimdCoord::from_f64(35.6, 139.7, 0.0, 1.0);
    let b = SimdCoord::from_f64(51.5, -0.1, 0.0, 2.0);

    c.bench_function("simd_distance", |bench| {
        bench.iter(|| black_box(a).distance(black_box(&b)))
    });
}

fn bench_batch_distances(c: &mut Criterion) {
    let coords: Vec<SimdCoord> = (0..1000)
        .map(|i| SimdCoord::from_f64(i as f64 * 0.1, i as f64 * 0.2, 0.0, 1.0))
        .collect();

    c.bench_function("batch_distances_1000", |bench| {
        bench.iter(|| batch_distances(black_box(&coords)))
    });
}

fn bench_find_nearest(c: &mut Criterion) {
    let coords: Vec<SimdCoord> = (0..1000)
        .map(|i| SimdCoord::from_f64(i as f64 * 0.1, i as f64 * 0.2, 0.0, 1.0))
        .collect();
    let query = SimdCoord::from_f64(50.0, 100.0, 0.0, 1.0);

    c.bench_function("find_nearest_k3_of_1000", |bench| {
        bench.iter(|| find_nearest(black_box(query), black_box(&coords), 3))
    });
}

fn bench_maglev_lookup(c: &mut Criterion) {
    let nodes: Vec<u64> = (0..100).collect();
    let maglev = MaglevHash::new(nodes);

    c.bench_function("maglev_lookup", |bench| {
        let mut key = 0u64;
        bench.iter(|| {
            key = key.wrapping_add(1);
            maglev.lookup(black_box(key))
        })
    });
}

fn bench_maglev_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("maglev_build");

    for n_nodes in [10, 50, 100] {
        let nodes: Vec<u64> = (0..n_nodes).collect();
        group.bench_with_input(
            BenchmarkId::new("nodes", n_nodes),
            &nodes,
            |b, nodes| {
                b.iter(|| MaglevHash::new(black_box(nodes.clone())))
            },
        );
    }
    group.finish();
}

fn bench_spatial_index(c: &mut Criterion) {
    let entries: Vec<SpatialEntry> = (0..1000u32)
        .map(|i| SpatialEntry {
            coord: SimdCoord::from_f64(i as f64 * 0.1, i as f64 * 0.2, 0.0, 1.0),
            node_id: i,
        })
        .collect();

    let index = SpatialIndex::build(entries);
    let query = SimdCoord::from_f64(50.0, 100.0, 0.0, 1.0);

    c.bench_function("spatial_nearest_k5_of_1000", |bench| {
        bench.iter(|| index.find_nearest_k(black_box(&query), 5))
    });
}

fn bench_vivaldi_predict_rtt(c: &mut Criterion) {
    let a = VivaldiCoord::at(35.6, 139.7, 0.0, 1.0);
    let b = VivaldiCoord::at(51.5, -0.1, 0.0, 2.0);

    c.bench_function("vivaldi_predict_rtt", |bench| {
        bench.iter(|| a.predict_rtt(black_box(&b)))
    });
}

criterion_group!(
    benches,
    bench_simd_distance,
    bench_batch_distances,
    bench_find_nearest,
    bench_maglev_lookup,
    bench_maglev_build,
    bench_spatial_index,
    bench_vivaldi_predict_rtt,
);
criterion_main!(benches);
