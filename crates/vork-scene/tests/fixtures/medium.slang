// Authored test medium: physical coefficients per metre, scaled by density
// and an optional exponential height falloff. Emission is independent of density.
import vork;
struct Inputs {
    // Absorption and scattering coefficients (m^-1) at density 1.
    // Omitted absorption leaves a scattering-only medium.
    [InputRange(0.0, 3.402823466e38, true, true)] float3 absorption = float3(0.0);
    [InputRange(0.0, 3.402823466e38, true, true)] float3 scattering = float3(0.5);
    // Multiplier on both coefficients.
    [InputRange(0.0, 3.402823466e38, true, true)] float density = 1.0;
    // Henyey-Greenstein phase asymmetry.
    [InputRange(-1.0, 1.0, true, true)] float anisotropy = 0.0;
    // Emitted radiance per metre, linear RGB.
    [InputRange(0.0, 3.402823466e38, true, true)] float3 emission = float3(0.0);
    // Inverse metres; density scales by exp(-height_falloff * medium-local y).
    [InputRange(0.0, 3.402823466e38, true, true)] float height_falloff = 0;
};
// Match the shipped authored height program: vacuum channels remain zero
// even when an unbounded exponential cannot be represented.
float scaled_coefficient(float coefficient, float scale) {
    return coefficient == 0.0 ? 0.0 : coefficient * scale;
}
float3 scaled_coefficients(float3 coefficient, float scale) {
    return float3(scaled_coefficient(coefficient.x, scale),
        scaled_coefficient(coefficient.y, scale), scaled_coefficient(coefficient.z, scale));
}
MediumCoefficients medium(MediumContext ctx, Inputs inputs) {
    float scale=inputs.density == 0.0 ? 0.0 : inputs.density*exp(-inputs.height_falloff*ctx.position.y);
    MediumCoefficients m;
    m.absorption=scaled_coefficients(inputs.absorption,scale);
    m.scattering=scaled_coefficients(inputs.scattering,scale);
    m.emission=inputs.emission;
    m.anisotropy=inputs.anisotropy;
    return m;
}
