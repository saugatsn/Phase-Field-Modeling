"""
Phase Field Fracture Analysis Validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime

def load_data(data_file):
    """Load and parse force-displacement data"""
    data = np.loadtxt(data_file)
    if data.shape[1] != 2:
        raise ValueError("Data file must contain exactly 2 columns (displacement, force)")
    
    displacement = data[:, 0]
    force = data[:, 1]
    print(f"Loaded {len(displacement)} data points from {data_file}")
    return displacement, force

def validate_physical_bounds(force):
    """
    Validate physical bounds - check for negative forces only
    """
    print("\nPHYSICAL BOUNDS VALIDATION")
    
    negative_forces = force < 0
    num_negative = np.sum(negative_forces)
    
    results = {
        'negative_force_count': num_negative,
        'negative_force_indices': np.where(negative_forces)[0].tolist(),
        'min_force': np.min(force),
        'passed': num_negative == 0
    }
    
    if num_negative > 0:
        print(f"WARNING: Found {num_negative} negative force values")
        print(f"Minimum force: {results['min_force']:.6f}")
        print(f"Indices: {results['negative_force_indices']}")
    else:
        print("All forces are non-negative (physically valid)")
    
    return results

def analyze_force_behavior(displacement, force):
    """
    Analyze force behavior and classify fracture type based on post-peak drop
    """
    print("\nFORCE BEHAVIOR ANALYSIS")
    
    # Find peak force and its index
    peak_idx = np.argmax(force)
    peak_force = force[peak_idx]
    peak_displacement = displacement[peak_idx]
    
    # Calculate post-peak force drop ratio
    final_force = np.mean(force[-5:])
    force_drop_ratio = (peak_force - final_force) / peak_force * 100
    
    # Classify fracture type
    if force_drop_ratio > 80:
        fracture_type = "Brittle"
    elif force_drop_ratio < 40:
        fracture_type = "Ductile"
    else:
        fracture_type = "Quasi-brittle"
    
    results = {
        'peak_force': peak_force,
        'peak_displacement': peak_displacement,
        'peak_index': peak_idx,
        'final_force': final_force,
        'force_drop_ratio': force_drop_ratio,
        'fracture_type': fracture_type
    }
    
    print(f"Peak force: {peak_force:.2f} at displacement {peak_displacement:.6f}")
    print(f"Final force: {final_force:.2f}")
    print(f"Force drop ratio: {force_drop_ratio:.1f}%")
    print(f"Fracture type: {fracture_type}")
    
    return results

def analyze_energy(displacement, force, peak_idx, peak_force, peak_displacement):
    """
    Analyze energy behavior with elastic work validation
    """
    print("\nENERGY ANALYSIS")
    
    # Calculate pre-peak (elastic) work using trapezoidal integration
    actual_elastic_work = np.trapz(force[:peak_idx+1], displacement[:peak_idx+1])
    
    # Theoretical linear elastic work (triangle estimate)
    theoretical_elastic_work = 0.5 * peak_force * peak_displacement
    
    # Calculate difference and validation
    work_difference = abs(actual_elastic_work - theoretical_elastic_work)
    work_error_percent = (work_difference / theoretical_elastic_work) * 100
    elastic_work_valid = work_error_percent <= 10.0
    
    # Calculate post-peak (fracture) work
    if peak_idx < len(force) - 1:
        fracture_work = np.trapz(force[peak_idx:], displacement[peak_idx:])
    else:
        fracture_work = 0
    
    # Total work
    total_work = np.trapz(force, displacement)
    
    results = {
        'actual_elastic_work': actual_elastic_work,
        'theoretical_elastic_work': theoretical_elastic_work,
        'work_difference': work_difference,
        'work_error_percent': work_error_percent,
        'elastic_work_valid': elastic_work_valid,
        'fracture_work': fracture_work,
        'total_work': total_work,
        'elastic_fraction': actual_elastic_work / total_work if total_work > 0 else 0
    }
    
    print(f"Actual elastic work: {actual_elastic_work:.6f}")
    print(f"Theoretical elastic work: {theoretical_elastic_work:.6f}")
    print(f"Difference: {work_difference:.6f} ({work_error_percent:.1f}%)")
    
    if elastic_work_valid:
        print("Elastic work validation PASSED (error <= 10%)")
    else:
        print("WARNING: Elastic work validation FAILED (error > 10%)")
    
    print(f"Post-peak fracture work: {fracture_work:.6f}")
    print(f"Total work: {total_work:.6f}")
    print(f"Elastic fraction: {results['elastic_fraction']:.1%}")
    
    return results

def additional_checks(force, peak_idx, peak_force):
    """
    Additional validation checks for force plateaus and incomplete fracture
    """
    print("\nADDITIONAL VALIDATION CHECKS")
    
    final_force = np.mean(force[-5:])
    post_peak_forces = force[peak_idx:]
    
    # Check for force plateau before maximum displacement
    plateau_threshold = 0.02
    has_plateau = False
    force_variation = 0
    
    if len(post_peak_forces) > 5:
        plateau_start_idx = max(1, int(0.8 * len(post_peak_forces)))
        plateau_forces = post_peak_forces[plateau_start_idx:]
        force_variation = (np.max(plateau_forces) - np.min(plateau_forces)) / np.mean(plateau_forces)
        has_plateau = force_variation < plateau_threshold and len(plateau_forces) > 3
    
    # Check for incomplete fracture (significant final force)
    final_force_ratio = final_force / peak_force
    incomplete_fracture = final_force_ratio > 0.1
    
    # Check for force oscillations in post-peak region
    oscillation_count = 0
    if len(post_peak_forces) > 10:
        peaks, _ = find_peaks(post_peak_forces, height=final_force*1.1)
        valleys, _ = find_peaks(-post_peak_forces)
        oscillation_count = len(peaks) + len(valleys)
    
    results = {
        'has_force_plateau': has_plateau,
        'plateau_force_variation': force_variation,
        'final_force_ratio': final_force_ratio,
        'incomplete_fracture': incomplete_fracture,
        'oscillation_count': oscillation_count
    }
    
    # Report findings
    if has_plateau:
        print(f"Force plateau detected before maximum displacement")
        print(f"Plateau variation: {force_variation:.1%}")
    else:
        print("No significant force plateau detected")
    
    if incomplete_fracture:
        print(f"WARNING: Possible incomplete fracture detected")
        print(f"Final force is {final_force_ratio:.1%} of peak force")
    else:
        print("Fracture appears complete (low final force)")
    
    if oscillation_count > 2:
        print(f"WARNING: {oscillation_count} force oscillations detected in post-peak region")
    else:
        print("No significant force oscillations detected")
    
    return results

def generate_validation_plot(displacement, force, results, save_path="validation_plot.png"):
    """
    Generate comprehensive validation plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    peak_idx = results['force_behavior']['peak_index']
    peak_force = results['force_behavior']['peak_force']
    peak_disp = results['force_behavior']['peak_displacement']
    
    # Main force-displacement curve
    ax1.plot(displacement, force, 'b-', linewidth=2, label='Force-Displacement')
    ax1.plot(displacement[peak_idx], force[peak_idx], 'ro', markersize=8, label='Peak Force')
    ax1.set_xlabel('Displacement')
    ax1.set_ylabel('Force')
    ax1.set_title('Force-Displacement Curve with Validation Markers')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Energy analysis
    ax2.fill_between(displacement[:peak_idx+1], 0, force[:peak_idx+1], 
                     alpha=0.3, color='blue', label='Elastic Work')
    
    if peak_idx < len(force) - 1:
        ax2.fill_between(displacement[peak_idx:], 0, force[peak_idx:], 
                         alpha=0.3, color='red', label='Fracture Work')
    
    ax2.plot([0, peak_disp, peak_disp], [0, peak_force, 0], 'k--', 
            linewidth=2, label='Theoretical Elastic')
    ax2.plot(displacement, force, 'k-', linewidth=1)
    ax2.set_xlabel('Displacement')
    ax2.set_ylabel('Force')
    ax2.set_title('Energy Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nValidation plot saved as {save_path}")

def generate_report(data_file, displacement, results, save_path="validation_report.txt"):
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE FIELD FRACTURE ANALYSIS - VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Data file: {data_file}")
    lines.append(f"Data points: {len(displacement)}")
    lines.append("")
    
    # Physical Bounds Section
    pb = results['physical_bounds']
    lines.append("PHYSICAL BOUNDS VALIDATION")
    lines.append(f"Status: {'PASSED' if pb['passed'] else 'FAILED'}")
    lines.append(f"Negative force points: {pb['negative_force_count']}")
    if pb['negative_force_count'] > 0:
        lines.append(f"Minimum force: {pb['min_force']:.6f}")
    lines.append("")
    
    # Force Behavior Section
    fb = results['force_behavior']
    lines.append("FORCE BEHAVIOR ANALYSIS")
    lines.append(f"Peak force: {fb['peak_force']:.6f}")
    lines.append(f"Peak displacement: {fb['peak_displacement']:.6f}")
    lines.append(f"Final force: {fb['final_force']:.6f}")
    lines.append(f"Force drop ratio: {fb['force_drop_ratio']:.1f}%")
    lines.append(f"Fracture type: {fb['fracture_type']}")
    lines.append("")
    
    # Energy Analysis Section
    en = results['energy']
    lines.append("ENERGY ANALYSIS")
    lines.append(f"Actual elastic work: {en['actual_elastic_work']:.6f}")
    lines.append(f"Theoretical elastic work: {en['theoretical_elastic_work']:.6f}")
    lines.append(f"Difference: {en['work_difference']:.6f}")
    lines.append(f"Error percentage: {en['work_error_percent']:.1f}%")
    lines.append(f"Validation status: {'PASSED' if en['elastic_work_valid'] else 'FAILED'}")
    lines.append(f"Fracture work: {en['fracture_work']:.6f}")
    lines.append(f"Total work: {en['total_work']:.6f}")
    lines.append(f"Elastic fraction: {en['elastic_fraction']:.1%}")
    lines.append("")
    
    # Additional Checks Section
    ad = results['additional']
    lines.append("ADDITIONAL VALIDATION CHECKS")
    lines.append(f"Force plateau detected: {'Yes' if ad['has_force_plateau'] else 'No'}")
    if ad['has_force_plateau']:
        lines.append(f"Plateau variation: {ad['plateau_force_variation']:.1%}")
    lines.append(f"Incomplete fracture: {'Yes' if ad['incomplete_fracture'] else 'No'}")
    if ad['incomplete_fracture']:
        lines.append(f"Final force ratio: {ad['final_force_ratio']:.1%}")
    lines.append(f"Force oscillations: {ad['oscillation_count']}")
    lines.append("")
    
    # Overall Summary
    lines.append("OVERALL VALIDATION SUMMARY")
    issues = []
    if pb['negative_force_count'] > 0:
        issues.append("Negative forces detected")
    if not en['elastic_work_valid']:
        issues.append("Elastic work validation failed")
    if ad['incomplete_fracture']:
        issues.append("Possible incomplete fracture")
    
    if issues:
        lines.append("ISSUES DETECTED:")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("No critical issues detected")
    
    lines.append("")
    lines.append("=" * 80)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"Validation report saved as {save_path}")

def run_validation(data_file="ForcevsDisp.txt"):
    """
    Run complete validation analysis
    """
    print("STARTING PHASE FIELD FRACTURE VALIDATION")

    # Load data
    displacement, force = load_data(data_file)
    
    # Run all validation checks
    results = {}
    results['physical_bounds'] = validate_physical_bounds(force)
    results['force_behavior'] = analyze_force_behavior(displacement, force)
    results['energy'] = analyze_energy(displacement, force, 
                                       results['force_behavior']['peak_index'],
                                       results['force_behavior']['peak_force'],
                                       results['force_behavior']['peak_displacement'])
    results['additional'] = additional_checks(force, 
                                             results['force_behavior']['peak_index'],
                                             results['force_behavior']['peak_force'])
    
    # Generate outputs
    generate_validation_plot(displacement, force, results)
    generate_report(data_file, displacement, results)
    
    print("VALIDATION COMPLETE")

    
    return results

# Run validation
run_validation("ForcevsDisp.txt")