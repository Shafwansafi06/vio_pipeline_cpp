// Regression test: FeatureDatabase must stay sorted by featid and free of
// dangling IDs after inserts, prune-by-timestamp, and prune-by-to_delete
// cycles. A dangling ID is one that get_feature()/find_feature_index() still
// resolves (or resolves to the wrong feature) after it should have been
// removed.
#include <iostream>

#include "../core/feature.hpp"

namespace {

bool sorted_and_consistent(const core::FeatureDatabase& db, const char* stage) {
    for (std::size_t i = 1; i < db.count; ++i) {
        if (db.features[i].featid <= db.features[i - 1].featid) {
            std::cerr << stage << ": features[] not strictly sorted by featid at index " << i << "\n";
            return false;
        }
    }
    // Every stored id must resolve to itself via the same lookup path used by
    // the estimator (find_feature_index / get_feature).
    for (std::size_t i = 0; i < db.count; ++i) {
        const int featid = db.features[i].featid;
        const int found = core::find_feature_index(db, featid);
        if (found != static_cast<int>(i)) {
            std::cerr << stage << ": find_feature_index(" << featid << ") returned " << found
                      << ", expected " << i << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    core::FeatureDatabase db;

    // Insert out of numeric order to exercise the sorted-insert path.
    const int ids[] = {50, 10, 30, 5, 40, 20};
    for (int id : ids) {
        core::update_feature(db, id, /*timestamp=*/1.0, /*cam_id=*/0, 0.0, 0.0, 0.0, 0.0);
    }
    if (db.count != 6 || !sorted_and_consistent(db, "after initial insert")) return 1;

    // Add more measurements at increasing timestamps to a subset of features.
    for (double t = 1.1; t < 2.0; t += 0.1) {
        core::update_feature(db, 10, t, 0, 0.0, 0.0, 0.0, 0.0);
        core::update_feature(db, 30, t, 1, 0.0, 0.0, 0.0, 0.0);
    }

    // Prune measurements older than 1.5s. Feature 5/20/40/50 only ever got the
    // t=1.0 measurement, so they must be dropped entirely (num_measurements
    // hits 0) while 10/30 survive with only their >1.5s measurements.
    core::cleanup_db_measurements(db, /*timestamp=*/1.5);
    if (!sorted_and_consistent(db, "after cleanup_db_measurements")) return 1;
    if (core::get_feature(db, 5) != nullptr || core::get_feature(db, 20) != nullptr ||
        core::get_feature(db, 40) != nullptr || core::get_feature(db, 50) != nullptr) {
        std::cerr << "feature pruned by cleanup_db_measurements is still resolvable (dangling ID)\n";
        return 1;
    }
    core::Feature* survivor = core::get_feature(db, 10);
    if (!survivor) {
        std::cerr << "feature 10 should have survived cleanup_db_measurements\n";
        return 1;
    }
    for (int m = 0; m < survivor->num_measurements; ++m) {
        if (survivor->measurements[m].timestamp <= 1.5) {
            std::cerr << "surviving feature retained a measurement at or before the cutoff\n";
            return 1;
        }
    }

    // Explicit to_delete marginalization path (used when a clone/landmark is
    // marginalized out of the EKF state): mark the surviving features and
    // confirm cleanup_db() removes them without corrupting sort order.
    core::Feature* f10 = core::get_feature(db, 10);
    f10->to_delete = true;
    core::cleanup_db(db);
    if (!sorted_and_consistent(db, "after cleanup_db")) return 1;
    if (core::get_feature(db, 10) != nullptr) {
        std::cerr << "feature marked to_delete is still resolvable after cleanup_db (dangling ID)\n";
        return 1;
    }
    // Feature 30 was never marked to_delete and must still be reachable.
    if (core::get_feature(db, 30) == nullptr) {
        std::cerr << "feature 30 was incorrectly removed by cleanup_db\n";
        return 1;
    }

    std::cout << "feature database sort/lookup invariants held across insert, "
                 "timestamp-prune, and to_delete-prune cycles\n";
    return 0;
}
