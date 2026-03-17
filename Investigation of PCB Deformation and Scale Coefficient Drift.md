# Technical Report

**Subject:** Investigation of PCB Deformation and Scale Coefficient Drift

## 1. Problem Overview

A deformation of the printed circuit board (PCB) has been observed in the module during testing. This deformation is associated with a **scale coefficient drift of approximately 3–5%**.

The deviation typically appears **after TVC or TVCh testing** (thermal vacuum cycling / thermal cycling) and then remains **stable during spacecraft operation**.

At present, the issue is mitigated operationally by **replacing the entire module**, rather than identifying and correcting the root cause.

The **underlying cause of the deformation and coefficient drift remains unknown**.

An internal assessment is currently underway to evaluate possible contributing factors. In parallel, there is discussion about **modifying the allowable tolerance**, potentially accepting the observed **3–5% deviation instead of the nominal 9% margin**, though this approach introduces technical risk.

---

## 2. Observed Facts

Based on the available information, the following points are confirmed:

• PCB deformation has been detected.
• A **scale coefficient drift of 3–5%** is observed.
• The drift appears **after TVC or TVCh testing**.
• Once the deviation appears, it remains **stable during spacecraft operation**.
• The **root cause is currently unknown**.
• The current mitigation method is **replacement of the entire module**.
• An investigation is underway to assess **possible contributing factors**.
• A proposal exists to **expand or adjust the allowable tolerance**, though this carries risk.

---

## 3. Preliminary Interpretation

The problem likely represents a **thermo-mechanical effect influencing electrical parameters** of the module.

If the scale coefficient changes after environmental testing and remains stable afterward, this suggests that **a permanent structural change occurs during the test conditions**.

This could indicate:

• residual mechanical stress in the PCB
• permanent deformation after thermal cycling
• changes in the geometry of sensitive components
• drift in analog circuitry parameters
• mechanical stress affecting electrical calibration

In short, the physical state of the system changes, and the electronics respond accordingly.

---

## 4. Potential Causes (Hypotheses)

These are working hypotheses that should be investigated experimentally.

### 4.1 Thermomechanical PCB deformation

Thermal vacuum or thermal cycling tests can induce **residual mechanical stress**, which may cause:

• PCB bending
• warping near sensitive components
• altered component alignment
• changes in mechanical reference geometry

If the system contains sensors or precision analog circuits, even small deformations can affect calibration.

---

### 4.2 Mismatch of thermal expansion coefficients

Different materials in the assembly expand differently under temperature changes.

Potentially involved materials:

• PCB substrate
• mechanical frame
• mounting hardware
• adhesives or potting compounds
• sensor packages

Repeated thermal cycling can cause **plastic deformation or accumulated stress**, shifting electrical characteristics.

---

### 4.3 Mounting and mechanical constraints

The deformation might not originate in the PCB itself but in the **mechanical mounting system**.

Possible issues:

• uneven mounting stress
• excessive torque on fasteners
• rigid mounting without thermal compensation
• misalignment of support points

Under thermal cycling these constraints may permanently warp the board.

---

### 4.4 Drift in electronic components

Scale coefficient changes may originate from **analog electronics** rather than mechanical deformation.

Possible sources:

• resistor network drift
• reference voltage instability
• amplifier parameter shifts
• microcracks in solder joints
• temperature-induced parameter aging

These issues could alter calibration after environmental stress.

---

### 4.5 Solder joint or interconnect damage

Thermal cycling can create **micro-fractures in solder joints or vias**, leading to:

• resistance changes
• altered signal paths
• gradual calibration shifts

These failures may not cause total malfunction but can introduce measurable parameter drift.

---

## 5. Current Mitigation Approach

The present operational solution is **replacement of the entire module** when the deviation occurs.

While this restores nominal performance, it does not address the root cause and therefore:

• does not prevent recurrence
• increases cost and maintenance effort
• does not improve system reliability understanding

---

## 6. Evaluation of Tolerance Expansion

A proposal exists to **expand the allowable tolerance range**, potentially accepting the observed **3–5% drift**.

### Potential Benefits

• reduction of rejected units
• simplified acceptance criteria
• reduced replacement requirements

### Risks

Accepting a deviation without understanding its cause may introduce significant risk:

• the deviation mechanism may evolve further during long-term operation
• the drift may correlate with other hidden degradation mechanisms
• the underlying production instability may remain undetected

In spacecraft systems, accepting unexplained parameter shifts can compromise **long-term reliability and mission safety**.

---

## 7. Required Additional Investigation

Further investigation is required to establish the **causal relationship between environmental testing, PCB deformation, and scale coefficient drift**.

Key investigation directions:

### 7.1 Mechanical analysis

• measure PCB flatness before and after testing
• quantify deformation magnitude
• analyze stress distribution at mounting points
• evaluate mounting torque effects

---

### 7.2 Electrical parameter localization

Identify which circuit block generates the coefficient drift:

• sensor element
• analog front end
• reference voltage circuitry
• ADC calibration stage
• power supply stability

---

### 7.3 Statistical data collection

Collect systematic data across multiple units:

• coefficient values before testing
• coefficient values after testing
• deformation measurements
• production batch information
• assembly conditions

This will allow identification of correlations.

---

### 7.4 Structural inspection of affected units

Defective units should undergo detailed inspection:

• microscopic inspection of solder joints
• PCB warpage measurement
• X-ray inspection of hidden joints
• layer integrity analysis

---

## 8. Conclusion

The observed **3–5% scale coefficient drift after TVC/TVCh testing** appears to be associated with **PCB deformation or thermomechanical effects within the module**.

The root cause has not yet been identified. Current mitigation through module replacement does not address the underlying mechanism.

Before modifying acceptance tolerances, it is essential to complete a **comprehensive investigation into the physical and electrical causes of the deviation** to ensure long-term reliability of the spacecraft system.

---

