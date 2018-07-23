import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import dataclasses

from cephlib.units import b2ssize


def get_data(rr: str, data: str) -> str:
    return re.search("(?ims)" + rr, data).group(0)  # type: ignore


@dataclasses.dataclass
class LSHWNetInfo:
    name: str
    speed: Optional[str]
    duplex: bool


@dataclasses.dataclass
class DiskInfo:
    size: int
    mount_point: Optional[str]
    device: Optional[str]


@dataclasses.dataclass
class CPUInfo:
    model: str
    cores: int


@dataclasses.dataclass
class HWInfo:
    hostname: Optional[str]
    cpu_info: List[CPUInfo]
    disks_info: Dict[str, DiskInfo]
    ram_size: int
    sys_name: Optional[str]
    mb: Optional[str]
    raw: str
    disks_raw_info: Dict[str, str]
    net_info: Dict[str, LSHWNetInfo]
    storage_controllers: List[str]

    def get_HDD_count(self):
        # SATA HDD COUNT, SAS 10k HDD COUNT, SAS SSD count, PCI-E SSD count
        return []

    def get_summary(self):
        cores = sum(count for _, count in self.cpu_info)
        disks = sum(info.size for info in self.disks_info.values())

        return {'cores': cores,
                'ram': self.ram_size,
                'storage': disks,
                'disk_count': len(self.disks_info)}

    def __str__(self):
        res = []

        summ = self.get_summary()
        summary = "Simmary: {cores} cores, {ram}B RAM, {disk}B storage"
        res.append(summary.format(cores=summ['cores'],
                                  ram=b2ssize(summ['ram']),
                                  disk=b2ssize(summ['storage'])))
        res.append(str(self.sys_name))
        if self.mb is not None:
            res.append("Motherboard: " + self.mb)

        if self.ram_size == 0:
            res.append("RAM: Failed to get RAM size")
        else:
            res.append(f"RAM {b2ssize(self.ram_size)}B")

        if self.cpu_info:
            res.append("CPU cores: Failed to get CPU info")
        else:
            res.append("CPU cores:")
            for name, count in self.cpu_info:
                res.append(f"    {count} * {name}" if count > 1 else f"    {name}")

        if self.storage_controllers:
            res.append("Disk controllers:")
            for descr in self.storage_controllers:
                res.append(f"    {descr}")

        if self.disks_info != {}:
            res.append("Storage devices:")
            for dev, info in sorted(self.disks_info.items()):
                res.append(f"    {dev} {b2ssize(info.size)}B {info.model}")
        else:
            res.append("Storage devices's: Failed to get info")

        if self.disks_raw_info != {}:
            res.append("Disks devices:")
            for dev, descr in sorted(self.disks_raw_info.items()):
                res.append(f"    {dev} {descr}")
        else:
            res.append("Disks devices's: Failed to get info")

        if self.net_info != {}:
            res.append("Net adapters:")
            for name, adapter in self.net_info.items():
                res.append(f"    {name} {adapter.is_phy} duplex={adapter.speed}")
        else:
            res.append("Net adapters: Failed to get net info")

        return str(self.hostname) + ":\n" + "\n".join("    " + i for i in res)


def parse_hw_info(lshw_out: str) -> Optional[HWInfo]:
    lshw_et = ET.fromstring(lshw_out)

    try:
        hostname = lshw_et.find("node").attrib['id']  # type: ignore
    except (AttributeError, KeyError):
        hostname = None

    try:
        sys_name = lshw_et.find("node/vendor").text + " " + lshw_et.find("node/product").text  # type: ignore
        sys_name = sys_name.lower().replace("(to be filled by o.e.m.)", "")
    except AttributeError:
        sys_name = None

    core = lshw_et.find("node/node[@id='core']")
    if core is None:
        return None

    try:
        mb: Optional[str] = " ".join(core.find(node).text for node in ['vendor', 'product', 'version'])  # type: ignore
    except AttributeError:
        mb = None

    cpu_info = []
    for cpu in core.findall("node[@class='processor']"):
        try:
            model = cpu.find('product').text  # type: ignore
            threads_node = cpu.find("configuration/setting[@id='threads']")
            cores = 1 if threads_node is None else int(threads_node.attrib['value'])  # type: ignore
        except (AttributeError, KeyError):
            pass
        else:
            assert isinstance(model, str)
            cpu_info.append(CPUInfo(model, cores))

    ram_size = 0
    for mem_node in core.findall(".//node[@class='memory']"):
        descr = mem_node.find('description')
        try:
            if descr is not None and descr.text == 'System Memory':
                mem_sz = mem_node.find('size')
                if mem_sz is None:
                    for slot_node in mem_node.find("node[@class='memory']"):  # type: ignore
                        slot_sz = slot_node.find('size')
                        if slot_sz is not None:
                            assert slot_sz.attrib['units'] == 'bytes'
                            ram_size += int(slot_sz.text)  # type: ignore
                else:
                    assert mem_sz.attrib['units'] == 'bytes'
                    ram_size += int(mem_sz.text)  # type: ignore
        except (AttributeError, KeyError, ValueError):
            pass

    net_info = {}
    for net in core.findall(".//node[@class='network']"):
        try:
            link = net.find("configuration/setting[@id='link']")
            if link.attrib['value'] == 'yes':  # type: ignore

                speed_node = net.find("configuration/setting[@id='speed']")
                speed = None if speed_node is None else speed_node.attrib['value']

                dup_node = net.find("configuration/setting[@id='duplex']")
                dup = False if dup_node is None else dup_node.attrib['value'] == "full"

                name = net.find("logicalname").text  # type: ignore
                assert isinstance(name, str)
                net_info[name] = LSHWNetInfo(name=name, speed=speed, duplex=dup)
        except (AttributeError, KeyError):
            pass

    storage_controllers = []
    for controller in core.findall(".//node[@class='storage']"):
        try:
            description = getattr(controller.find("description"), 'text', "")
            product = getattr(controller.find("product"), 'text', "")
            vendor = getattr(controller.find("vendor"), 'text', "")
            dev = getattr(controller.find("logicalname"), 'text', "")
            storage_controllers.append((f"{dev}: " if dev else "") + f"{description} {vendor} {product}")
        except AttributeError:
            pass

    disks_raw_info: Dict[str, str] = {}
    disks_info = {}
    for disk in core.findall(".//node[@class='disk']"):
        try:
            lname_node = disk.find('logicalname')
            if lname_node is not None:

                dev = lname_node.text.split('/')[-1]  # type: ignore
                if dev == "" or dev[-1].isdigit():
                    continue

                sz_node = disk.find('size')
                assert sz_node.attrib['units'] == 'bytes'  # type: ignore
                sz = int(sz_node.text)  # type: ignore
                disks_info[dev] = DiskInfo(size=sz, mount_point=None, device=None)
            else:
                description = disk.find('description').text  # type: ignore
                product = disk.find('product').text  # type: ignore
                vendor = disk.find('vendor').text  # type: ignore
                version = disk.find('version').text  # type: ignore
                serial = disk.find('serial').text  # type: ignore
                businfo = disk.find('businfo').text  # type: ignore
                assert isinstance(businfo, str)
                disks_raw_info[businfo] = f"{description} {product} {vendor} {version} {serial}"
        except (AttributeError, KeyError):
            pass

    return HWInfo(raw=lshw_out,
                  disks_raw_info=disks_raw_info,
                  disks_info=disks_info,
                  sys_name=sys_name,
                  hostname=hostname,
                  mb=mb,
                  cpu_info=cpu_info,
                  storage_controllers=storage_controllers,
                  net_info=net_info,
                  ram_size=ram_size)


def get_dev_file_name(path_or_name: str) -> str:
    res = path_or_name[len("/dev/"):] if path_or_name.startswith('/dev/') else path_or_name
    assert '/' not in res
    return res