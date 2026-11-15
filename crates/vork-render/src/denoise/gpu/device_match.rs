//! Match only identities shared by the rendering and denoising APIs.
pub(super) fn matches(kind: i32, renderer_uuid: &[u8; 16], renderer_pci: Option<[u32; 4]>,
    uuid: Option<&[u8; 16]>, pci: Option<[u32; 4]>) -> bool {
    let _ = (renderer_pci, pci);
    match kind {
        3 => uuid == Some(renderer_uuid),
        #[cfg(target_os = "linux")]
        4 => renderer_pci.is_some() && renderer_pci == pci,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_requires_the_vulkan_uuid() {
        assert!(matches(3, &[1; 16], None, Some(&[1; 16]), None));
        assert!(!matches(3, &[1; 16], Some([0, 3, 0, 0]), Some(&[2; 16]), Some([0, 3, 0, 0])));
        assert!(!matches(3, &[1; 16], None, None, None));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hip_uses_pci_even_when_radv_and_hip_uuids_differ() {
        let address = [0, 3, 0, 0];
        assert!(matches(4, &[1; 16], Some(address), Some(&[2; 16]), Some(address)));
        // Every PCI component participates: another GPU must never be chosen.
        for component in 0..4 {
            let mut other = address;
            other[component] += 1;
            assert!(!matches(4, &[1; 16], Some(address), Some(&[1; 16]), Some(other)));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hip_requires_both_apis_to_report_pci_support() {
        for (renderer, oidn) in [(None, None), (None, Some([0; 4])), (Some([0; 4]), None)] {
            assert!(!matches(4, &[1; 16], renderer, Some(&[1; 16]), oidn));
        }
        // Address availability comes from advertised support, not a sentinel.
        assert!(matches(4, &[1; 16], Some([0; 4]), None, Some([0; 4])));
        assert!(!matches(1, &[1; 16], Some([0; 4]), Some(&[1; 16]), Some([0; 4])));
    }
}
