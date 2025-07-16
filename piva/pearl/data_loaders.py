import datetime
import h5py
import numpy as np
from pathlib import Path

from piva.data_loaders import Dataloader

class DataloaderPEARL(Dataloader):
    """
    PEARL data files
    """

    name = "PEARL"

    def __init__(self):
        super(DataloaderPEARL, self).__init__()

    def load_data(self, filename, metadata=False):
        """
        Recognize correct format and load data from the file.

        :param filename: absolute path to the file
        :param metadata: if :py:obj:`True`, read only metadata and size of the
                         dataset. Not used here, but required to mach format
                         of other **Dataloaders**. See :meth:`load_ses_zip`
                         for more info.
        :return: loaded dataset with available metadata
        """

        path = Path(filename)
        if path.suffix == ".h5" and path.stem.startswith("pshell"):
            self.load_pshell(filename, metadata=metadata)
        else:
            raise NotImplementedError

        return self.validate_at_return(filename)

    def _find_scan(self, h5):
        try:
            scan = h5['scan1']
        except KeyError:
            scan = h5['scan 1']
        
        return scan

    def _find_region(self, scan):
        try:
            region = scan['region1']
        except KeyError:
            try:
                region = scan['region 1']
            except KeyError:
                region = scan

        return region

    def _find_scienta_image(self, scan):
        try:
            image = scan['ScientaImage']
        except KeyError:
            image = scan['Scienta image']

        return image

    def load_pshell(self, filename, metadata=False):
        h5 = h5py.File(filename, "r")

        scan = self._find_scan(h5)
        scan = self._find_region(scan)

        writables = [s.decode() for s in scan.attrs['Writables']]
        readables = [s.decode() for s in scan.attrs['Readables']]
        attrs = scan['attrs']

        # original dimensions: 0: angle/pos, 1: energy, 2: scan
        # imported dimensions: 0: scan, 1: angle/pos, 2: energy
        data = self._find_scienta_image(scan)
        shape = data.shape

        if metadata:
            if len(data.shape) == 2:
                new_shape = (0, shape[0], shape[1])
            elif len(data.shape) == 3:
                new_shape = (shape[2], shape[0], shape[1])
            else:
                raise NotImplementedError

            self.ds.data = np.zeros(new_shape)
            self.ds.xscale = np.zeros((new_shape[0],))
            self.ds.yscale = np.zeros((new_shape[1],))
            self.ds.zscale = np.zeros((new_shape[2],))
        else:
            if len(data.shape) == 2:
                self.ds.data = data[None, :]
            elif len(data.shape) == 3:
                self.ds.data = np.transpose(data, (2, 0, 1))
            else:
                raise NotImplementedError

            scale_name = writables[0]
            self.ds.xscale = scan[scale_name]
            for scale_name in writables:
                scale = scan[scale_name]
                if scale.min() < scale.max():
                    self.ds.xscale = scale
                    break

            self.ds.yscale = scan['ScientaSlices']
            self.ds.zscale = scan['ScientaChannels']

        self.ds.x = np.mean(attrs['ManipulatorX'])
        self.ds.y = np.mean(attrs['ManipulatorY'])
        self.ds.z = np.mean(attrs['ManipulatorZ'])
        self.ds.theta = np.mean(attrs['ManipulatorTheta'])
        self.ds.tilt = np.mean(attrs['ManipulatorTilt'])
        self.ds.phi = np.mean(attrs['ManipulatorPhi'])
        self.ds.temp = np.mean(attrs['ManipulatorTempA'])
        self.ds.pressure = np.mean(attrs['ChamberPressure'])
        self.ds.hv = np.mean(attrs['MonoEnergy'])
        self.ds.wf = 4.6
        self.ds.Ef = 0.0

        self.ds.polarization = 'LH'
        self.ds.PE = np.mean(np.asarray([int(pe.decode()) for pe in attrs['PassEnergy']]))
        self.ds.exit_slit = np.mean(attrs['ExitSlit'])
        self.ds.FE = np.mean(attrs['FrontendVSize'])

        self.ds.acq_mode = attrs['AcquisitionMode'][0].decode()
        self.ds.lens_mode = attrs['LensMode'][0].decode()
        # self.ds.ana_slit
        # self.ds.defl_angle
        self.ds.n_sweeps = np.mean(attrs['NumIterations'])
        self.ds.DT = np.mean(attrs['ScientaDwellTime'])
        self.ds.date = datetime.datetime.fromtimestamp(scan.attrs['Start'] / 1000).isoformat()

        if self.ds.xscale.size == 1:
            self.ds.scan_type = "cut"
        else:
            self.ds.scan_type = ",".join(writables)
            self.ds.scan_dim = [
                self.ds.xscale[0],
                self.ds.xscale[-1],
                np.abs(self.ds.xscale[0] - self.ds.xscale[1]),
            ]
