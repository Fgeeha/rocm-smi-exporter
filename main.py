#!/usr/bin/env python3
import json
import logging
import re
import time
from subprocess import check_output, CalledProcessError
from prometheus_client import (
    start_http_server,
    Gauge,
    REGISTRY,
    PROCESS_COLLECTOR,
    PLATFORM_COLLECTOR,
)

PORT_L = 9101


logger = logging.getLogger("rocm_smi_exporter")
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Отключаем дефолтные метрики Python, чтобы не шумели
try:
    REGISTRY.unregister(PROCESS_COLLECTOR)
    REGISTRY.unregister(PLATFORM_COLLECTOR)
    # gc может называться по-разному в версиях lib; не критично, если не найдём.
    gc_name = "python_gc_objects_collected_total"
    if gc_name in REGISTRY._names_to_collectors:
        REGISTRY.unregister(REGISTRY._names_to_collectors[gc_name])
except Exception as e:
    logger.debug(f"Skip unregister defaults: {e}")

# ========= Prometheus Gauges =========
gpuEdgeTemperature = Gauge(
    "rocm_smi_edge_temperature_celsius",
    "Edge temperature (°C)",
    ["device_name", "device_id", "subsystem_id"],
)
gpuJunctionTemperature = Gauge(
    "rocm_smi_hotspot_temperature_celsius",
    "Hotspot/junction temperature (°C)",
    ["device_name", "device_id", "subsystem_id"],
)
gpuMemTemperature = Gauge(
    "rocm_smi_memory_temperature_celsius",
    "Memory temperature (°C)",
    ["device_name", "device_id", "subsystem_id"],
)

gpuUsage = Gauge(
    "rocm_smi_gpu_usage_percent",
    "GPU usage (%)",
    ["device_name", "device_id", "subsystem_id"],
)
gpuVRAMUsage = Gauge(
    "rocm_smi_vram_allocation_percent",
    "VRAM allocation (%)",
    ["device_name", "device_id", "subsystem_id"],
)
umcActivity = Gauge(
    "rocm_smi_umc_activity_percent",
    "UMC (memory controller) activity (%)",
    ["device_name", "device_id", "subsystem_id"],
)
mmActivity = Gauge(
    "rocm_smi_mm_activity_percent",
    "Multimedia activity (%)",
    ["device_name", "device_id", "subsystem_id"],
)

socketPowerNow = Gauge(
    "rocm_smi_socket_power_watts",
    "Current socket power (W)",
    ["device_name", "device_id", "subsystem_id"],
)
socketPowerAvg = Gauge(
    "rocm_smi_socket_power_average_watts",
    "Average socket power (W)",
    ["device_name", "device_id", "subsystem_id"],
)
pkgPowerAvg = Gauge(
    "rocm_smi_gfx_package_power_average_watts",
    "Average graphics package power (W)",
    ["device_name", "device_id", "subsystem_id"],
)
pkgPowerMax = Gauge(
    "rocm_smi_gfx_package_power_max_watts",
    "Max graphics package power (W)",
    ["device_name", "device_id", "subsystem_id"],
)

voltageMilliV = Gauge(
    "rocm_smi_voltage_millivolt",
    "Voltage (mV)",
    ["device_name", "device_id", "subsystem_id", "rail"],
)  # rail: soc/gfx/mem
fanRpm = Gauge(
    "rocm_smi_fan_speed_rpm",
    "Fan speed (RPM)",
    ["device_name", "device_id", "subsystem_id"],
)

clkCurrentMHz = Gauge(
    "rocm_smi_clock_current_mhz",
    "Current clock (MHz)",
    ["device_name", "device_id", "subsystem_id", "clock"],
)  # clock: gfxclk/socclk/uclk/vclk0/dclk0
clkAverageMHz = Gauge(
    "rocm_smi_clock_average_mhz",
    "Average clock (MHz)",
    ["device_name", "device_id", "subsystem_id", "clock"],
)

pcieWidthLanes = Gauge(
    "rocm_smi_pcie_link_width_lanes",
    "PCIe link width (lanes)",
    ["device_name", "device_id", "subsystem_id"],
)
pcieSpeedGTs = Gauge(
    "rocm_smi_pcie_link_speed_gtps",
    "PCIe link speed (GT/s)",
    ["device_name", "device_id", "subsystem_id"],
)

energyAccumulatorUJ = Gauge(
    "rocm_smi_energy_accumulator_uj",
    "Energy accumulator (µJ units per header)",
    ["device_name", "device_id", "subsystem_id"],
)
accumulatedEnergyUJ = Gauge(
    "rocm_smi_accumulated_energy_uj",
    "Accumulated energy (µJ)",
    ["device_name", "device_id", "subsystem_id"],
)

# Информация о девайсе и версиях как gauge=1 с лейблами (info-метрика)
deviceInfo = Gauge(
    "rocm_smi_device_info",
    "Static device info",
    [
        "device_name",
        "device_id",
        "subsystem_id",
        "vbios",
        "gfx_version",
        "card_series",
        "card_vendor",
    ],
)
softwareInfo = Gauge(
    "rocm_smi_software_info",
    "Software versions",
    ["driver_version", "rocm_smi_version", "rocm_smi_lib_version"],
)

# ========== Helpers ==========
_rx_number = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _is_na(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return False
    return str(v).strip().upper() == "N/A"


def _to_float(v, *, scale=1.0):
    """Достаёт первое число из строки/числа, применяет scale; для 'N/A' -> None."""
    if _is_na(v):
        return None
    if isinstance(v, (int, float)):
        return float(v) * scale
    s = str(v)
    m = _rx_number.search(s)
    if not m:
        return None
    return float(m.group(0)) * scale


def _first_key(d: dict, keys: list[str]):
    """Вернёт первое существующее имя ключа из списка."""
    for k in keys:
        if k in d:
            return k
    return None


def _set_if_not_none(gauge: Gauge, labels: dict, value):
    if value is None:
        return
    try:
        gauge.labels(**labels).set(float(value))
    except Exception as e:
        logger.debug(f"skip set {gauge._name}: {e}")


def _get_versions():
    rsmi_ver = ""
    rlib_ver = ""
    try:
        out = check_output(
            ["rocm-smi", "-V"], text=True, encoding="utf-8", errors="ignore"
        )
        m1 = re.search(r"ROCM-SMI version:\s*(.+)", out)
        m2 = re.search(r"ROCM-SMI-LIB version:\s*(.+)", out)
        rsmi_ver = m1.group(1).strip() if m1 else ""
        rlib_ver = m2.group(1).strip() if m2 else ""
    except CalledProcessError as e:
        logger.warning(f"rocm-smi -V failed: {e}")
    return rsmi_ver, rlib_ver


def getGPUMetrics():
    metrics = json.loads(check_output(["rocm-smi", "-a", "--json"]))
    logger.info("[X] Retrieved metrics from rocm-smi.")
    return metrics


# ========== Main ==========
if __name__ == "__main__":
    start_http_server(PORT_L)
    logger.info(f"[X] Started http server on port {PORT_L}...")

    rsmi_ver, rlib_ver = _get_versions()
    driver_version = ""  # из блока 'system' ниже заполним при первом проходе
    # Раз в 10 секунд обновляем метрики
    while True:
        data = getGPUMetrics()

        # system — вне карточек
        if "system" in data and "Driver version" in data["system"]:
            driver_version = str(data["system"]["Driver version"]).strip()
            softwareInfo.labels(
                driver_version=driver_version,
                rocm_smi_version=rsmi_ver,
                rocm_smi_lib_version=rlib_ver,
            ).set(1)

        for card, m in data.items():
            if card == "system":
                continue

            labels = {
                "device_name": m.get("Device Name", "unknown"),
                "device_id": m.get("Device ID", "unknown"),
                "subsystem_id": m.get("Subsystem ID", "unknown"),
            }

            # Static info (1 раз можно, но set() идемпотентен)
            deviceInfo.labels(
                device_name=labels["device_name"],
                device_id=labels["device_id"],
                subsystem_id=labels["subsystem_id"],
                vbios=m.get("VBIOS version", ""),
                gfx_version=m.get("GFX Version", "")
                or m.get("GFX version", ""),
                card_series=m.get("Card Series", ""),
                card_vendor=m.get("Card Vendor", ""),
            ).set(1)

            # Temperatures (edge/junction/memory): поддерживаем старые и новые ключи
            _set_if_not_none(
                gpuEdgeTemperature,
                labels,
                _to_float(
                    m.get(
                        _first_key(
                            m,
                            [
                                "Temperature (Sensor edge) (C)",
                                "temperature_edge (C)",
                            ],
                        )
                    )
                ),
            )
            _set_if_not_none(
                gpuJunctionTemperature,
                labels,
                _to_float(
                    m.get(
                        _first_key(
                            m,
                            [
                                "Temperature (Sensor junction) (C)",
                                "temperature_hotspot (C)",
                            ],
                        )
                    )
                ),
            )
            _set_if_not_none(
                gpuMemTemperature,
                labels,
                _to_float(
                    m.get(
                        _first_key(
                            m,
                            [
                                "Temperature (Sensor memory) (C)",
                                "temperature_mem (C)",
                            ],
                        )
                    )
                ),
            )

            # Usage
            usage = _to_float(
                m.get(
                    _first_key(m, ["GPU use (%)", "average_gfx_activity (%)"])
                )
            )
            _set_if_not_none(gpuUsage, labels, usage)

            vram = _to_float(m.get("GPU Memory Allocated (VRAM%)"))
            _set_if_not_none(gpuVRAMUsage, labels, vram)

            _set_if_not_none(
                umcActivity,
                labels,
                _to_float(m.get("average_umc_activity (%)")),
            )
            _set_if_not_none(
                mmActivity, labels, _to_float(m.get("average_mm_activity (%)"))
            )

            # Power (разные имена в разных версиях)
            now_power_key = _first_key(
                m,
                [
                    "Current Socket Graphics Package Power (W)",
                    "current_socket_power (W)",
                ],
            )
            avg_sock_key = _first_key(
                m, ["average_socket_power (W)"]
            )  # новое имя
            avg_pkg_key = _first_key(
                m, ["Average Graphics Package Power (W)"]
            )  # старое имя
            max_pkg_key = _first_key(m, ["Max Graphics Package Power (W)"])

            _set_if_not_none(
                socketPowerNow, labels, _to_float(m.get(now_power_key))
            )
            _set_if_not_none(
                socketPowerAvg, labels, _to_float(m.get(avg_sock_key))
            )
            _set_if_not_none(
                pkgPowerAvg, labels, _to_float(m.get(avg_pkg_key))
            )
            _set_if_not_none(
                pkgPowerMax, labels, _to_float(m.get(max_pkg_key))
            )

            # Voltages
            for rail_key, rail_name in [
                ("voltage_soc (mV)", "soc"),
                ("voltage_gfx (mV)", "gfx"),
                ("voltage_mem (mV)", "mem"),
                ("Voltage (mV)", "vcore"),
            ]:
                v = _to_float(m.get(rail_key))
                if v is not None:
                    voltageMilliV.labels(**labels, rail=rail_name).set(v)

            # Fan
            _set_if_not_none(
                fanRpm, labels, _to_float(m.get("current_fan_speed (rpm)"))
            )

            # Clocks (current)
            for key, cname in [
                ("current_gfxclk (MHz)", "gfxclk"),
                ("current_socclk (MHz)", "socclk"),
                ("current_uclk (MHz)", "uclk"),
                ("current_vclk0 (MHz)", "vclk0"),
                ("current_dclk0 (MHz)", "dclk0"),
            ]:
                val = _to_float(m.get(key))
                if val is not None:
                    clkCurrentMHz.labels(**labels, clock=cname).set(val)

            # Clocks (average)
            for key, cname in [
                ("average_gfxclk_frequency (MHz)", "gfxclk"),
                ("average_socclk_frequency (MHz)", "socclk"),
                ("average_uclk_frequency (MHz)", "uclk"),
                ("average_vclk0_frequency (MHz)", "vclk0"),
                ("average_dclk0_frequency (MHz)", "dclk0"),
            ]:
                val = _to_float(m.get(key))
                if val is not None:
                    clkAverageMHz.labels(**labels, clock=cname).set(val)

            # PCIe: width lanes и speed (в JSON — «0.1 GT/s», т.е. 80 => 8.0 GT/s)
            width = _to_float(m.get("pcie_link_width (Lanes)"))
            if width is not None:
                pcieWidthLanes.labels(**labels).set(width)
            speed_01 = _to_float(m.get("pcie_link_speed (0.1 GT/s)"))
            if speed_01 is not None:
                pcieSpeedGTs.labels(**labels).set(speed_01 * 0.1)

            # Энергия
            _set_if_not_none(
                energyAccumulatorUJ,
                labels,
                _to_float(
                    m.get(
                        _first_key(
                            m,
                            [
                                "energy_accumulator (15.259uJ (2^-16))",
                                "Energy counter",
                            ],
                        )
                    )
                ),
            )
            _set_if_not_none(
                accumulatedEnergyUJ,
                labels,
                _to_float(m.get("Accumulated Energy (uJ)")),
            )

        logger.info("[X] Refreshed GPU metrics.")
        time.sleep(10)
